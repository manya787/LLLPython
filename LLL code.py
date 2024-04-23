import numpy as np
import heapq
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

# Geodesic function from the first prompt
def geodesic(graph, start):
    priority_queue = [(0, start)]
    visited = set()
    distances = {start: 0}
    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)
        if current_node in visited:
            continue
        visited.add(current_node)
        for neighbor, weight in graph[current_node].items():
            distance = current_distance + weight
            if distance < distances.get(neighbor, float('inf')):
                distances[neighbor] = distance
                heapq.heappush(priority_queue, (distance, neighbor))
    return distances

# Rest of the code remains the same...

#only to find the eucledian distance bwetween two points
def sqdist(X, Y=None, w=None):
    if Y is None:
        Y = X
    if w is None:
        w = np.ones(X.shape[1])
    X = X * np.sqrt(w)
    Y = Y * np.sqrt(w)
    x = np.sum(X ** 2, axis=1)
    y = np.sum(Y ** 2, axis=1)
    sqd = np.maximum(np.add.outer(x, y) - 2 * X.dot(Y.T), 0)
    return sqd

#dividing the dataset into k clusters around k centroids ( randomly chosen )
def kmeans(X, K):
    kmeans = KMeans(n_clusters=K, init='random').fit(X)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    return centers, labels

#The function returns the selected landmark points Y0.
def lmarks(Y, L, method='random'):
    if method == 'random':
        idx = np.random.choice(Y.shape[0], L, replace=False)
        Y0 = Y[idx]
    elif method == 'kmeans':
        centers, _ = kmeans(Y, L)
        Y0 = centers
    return Y0

#These weights=contibution
#determine the contribution of each nearest landmark point to the reconstruction of the data point.
def lweights(Y, Y0, kZ):
    dist = cdist(Y, Y0)
    idx = np.argsort(dist, axis=1)[:, :kZ]
    T = np.zeros((Y.shape[0], Y0.shape[0]))
    for i in range(Y.shape[0]):
        dY = Y[i] - Y0[idx[i]]
        C = np.dot(dY, dY.T)
        if Y0.shape[0] > Y0.shape[1]:
            C += np.eye(kZ) * 1e-10 * np.sum(np.diag(C))
        zi = np.linalg.solve(C, np.ones(kZ))
        zi = zi / np.sum(zi)
        T[i, idx[i]] = zi
    return T.T

#Output:
#X: The projected data points onto the latent space, representing the low-dimensional embeddings.
#Y0: The selected landmarks from the original data.
#X0: The eigenvectors of the Laplacian corresponding to the smallest eigenvalues.
def lll(Y, W, d, L, kZ=None, nl=None):
    if kZ is None:
        kZ = d + 1
    if nl is None:
        nl = 0
    Y0 = lmarks(Y, L)
    Z = lweights(Y, Y0, kZ)
    D = np.sum(W, axis=1)
    L = np.diag(D) - W
    if nl == 1:
        DD = np.diag(D ** -0.5)
        L = DD.dot(L).dot(DD)
    LL = (L + L.T) / 2
    evals, U = np.linalg.eigh(LL)
    idx = np.argsort(evals)[1:d+1]
    X0 = U[:, idx]
    X = np.dot(Z, X0)  # Transpose Z here
    return X, Y0, X0


# used to add new set of data points directly in low dimension(out of sample)
def lllmap(Yt, Y0, kZ, X0):
    Zt = lweights(Yt, Y0, kZ)
    Xt = np.dot(Zt, X0)  # Transpose Zt here
    return Xt


def gaussaff(Y, params):
    method = params[0]
    if method == 'k':
        K = params[1]
        s = params[2]
        return np.exp(-cdist(Y, Y, 'sqeuclidean') / (2 * s ** 2))
    elif method == 'f':
        s = params[1]
        return np.exp(-cdist(Y, Y, 'sqeuclidean') / (2 * s ** 2))
    elif method == 'b':
        eps = params[1]
        return np.exp(-cdist(Y, Y, 'sqeuclidean') / (2 * eps))

# Function to perform Helmert transformation
def helmert_transformation(Xloc, Xglo):
    n = Xloc.shape[0]
    Xloc_centroid = np.mean(Xloc, axis=0)
    Xglo_centroid = np.mean(Xglo, axis=0)
    Xloc_shifted = Xloc - Xloc_centroid
    Xglo_shifted = Xglo - Xglo_centroid
    # Calculate rotation matrix
    Rm = np.dot(Xglo_shifted.T, Xloc_shifted)
    # Handle potential non-convergence of SVD
    try:
        U, _, Vt = np.linalg.svd(Rm)
    except np.linalg.LinAlgError:
        # Add a small perturbation to the diagonal of Rm
        Rm_perturbed = Rm + np.eye(Rm.shape[0]) * 1e-6
        U, _, Vt = np.linalg.svd(Rm_perturbed)
    R = np.dot(U, Vt)
    # Calculate scaling factor
    s = np.linalg.norm(Xglo_shifted) / np.linalg.norm(Xloc_shifted)
    # Calculate translation
    t = Xglo_centroid - s * np.dot(R, Xloc_centroid)
    return R, s, t


# Function to apply Helmert transformation
def apply_helmert_transformation(Xloc, R, s, t):
    Xglo = s * np.dot(Xloc, R.T) + t
    return Xglo

def demo():
    np.random.seed(0)
    N = 10
    x = np.random.rand(N)
    y = np.random.rand(N)
    Y = np.column_stack((x, y))

    # Compute Gaussian affinity matrix
    W = gaussaff(Y, ['k', 45, 0.005])

    # Run LLL algorithm
    d = 2
    if len(Y) < d:
        d = len(Y)  # Ensure d is not greater than the number of data points
    L = 5
    X, Y0, X0 = lll(Y, W, d, L)

    # Run out-of-sample mapping
    Xt = lllmap(Y, Y0, d+1, X0)

    # Apply Helmert transformation to embedded data
    R, s, t = helmert_transformation(X, Xt)

    # Apply Helmert transformation to out-of-sample data
    Xt_transformed = apply_helmert_transformation(Xt, R, s, t)

    # Plot the original data
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.title('Original Data')
    plt.scatter(Y[:, 0], Y[:, 1], c='b', marker='o', s=10)

    # Plot the embedded data
    plt.subplot(1, 3, 2)
    plt.title('Embedded Data (LLL)')
    plt.scatter(X[:, 0], X[:, 1], c='r', marker='o', s=50)
    plt.scatter(Y0[:, 0], Y0[:, 1], c='g', marker='o', s=40, alpha=0.5, label='Landmarks')
    plt.legend()

    # Plot the transformed out-of-sample data
    plt.subplot(1, 3, 3)
    plt.scatter(Xt[:, 0], Xt[:, 1], c='b', marker='x', s=10, label='Out-of-sample')
    plt.scatter(Xt_transformed[:, 0], Xt_transformed[:, 1], c='m', marker='x', s=10, label='Transformed Out-of-sample')
    plt.scatter(Xt_transformed[:, 0], Xt_transformed[:, 1], c='m', marker='x', s=10, label='Transformed Points')
    plt.legend()
    plt.title('Transformed Out-of-sample Data', loc='left', pad=20)  # Move title outside
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust layout to make space for title
    plt.show()
    print("Input Data Points:")
    print(Y)
    print("\nOutput Data Points (Embedded):")
    print(X)
    print("\nLocal Landmark Points:")
    print(Y0)
    print("\nEstimated Points:")
    print(Xt_transformed)  # Print the transformed out-of-sample data
if __name__ == '__main__':
    demo()