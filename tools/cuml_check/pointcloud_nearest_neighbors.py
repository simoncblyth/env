import numpy as np
from cuml.neighbors import NearestNeighbors
import cudf


def make_circle_2D(n_points = 1000, center=np.array([0,0], dtype=np.float32), radius=1):
    u = np.random.rand(n_points)
    phi = 2.*np.pi*u

    pc = np.zeros( (n_points, 2), dtype=np.float32 )
    pc[:,0] = radius*np.cos(phi)
    pc[:,1] = radius*np.sin(phi)
    pc += center

    nr = np.zeros( (n_points, 2), dtype=np.float32 )
    nr[:,0] = np.cos(phi)
    nr[:,1] = np.sin(phi)
    return pc, nr



# Generate two random point clouds (e.g., 2D points)
n_points1 = 200
n_points2 = 100

pc1,nr1 = make_circle_2D( n_points = n_points1, center=np.array([-0.9,0]))
pc2,nr2 = make_circle_2D( n_points = n_points2, center=np.array([+0.9,0]))


# Convert numpy arrays to cuDF DataFrames (cuML works with cuDF for GPU acceleration)
pc1_cudf = cudf.DataFrame(pc1, columns=['x', 'y'])
pc2_cudf = cudf.DataFrame(pc2, columns=['x', 'y'])

# Initialize the NearestNeighbors model
# n_neighbors=1 to find the single nearest neighbor for each point
nn_model = NearestNeighbors(n_neighbors=1, metric='euclidean')

# Fit the model on the first point cloud
nn_model.fit(pc1_cudf)

# Find the nearest neighbors in pc1 for each point in pc2
distances, indices = nn_model.kneighbors(pc2_cudf)

# Convert results to numpy arrays for easier handling
distances = distances.to_numpy()
indices = indices.to_numpy()

# Print some example results


print("Nearest neighbor pairs (point in cloud2 -> nearest point in cloud1):")
for i in range(5):  # Print first 5 pairs for brevity
    print(f"Point {i} in cloud2: {pc2[i]}, Nearest in cloud1: {pc1[indices[i]]}, Distance: {distances[i]}")

# Optional: Visualize the point clouds and nearest neighbor pairs
import matplotlib.pyplot as plt

plt.scatter(pc1[:, 0], pc1[:, 1], c='blue', label='Point Cloud 1', alpha=0.5)
plt.scatter(pc2[:, 0], pc2[:, 1], c='red', label='Point Cloud 2', alpha=0.5)





# Plot lines connecting nearest neighbors
for i in range(min(100, n_points2)):
    p2 = pc2[i]
    p1 = pc1[indices[i]]
    plt.plot([p2[0], p1[0]], [p2[1], p1[1]], 'k-', alpha=0.2)

plt.legend()
plt.title("Nearest Neighbor Pairs Between Two Point Clouds")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()


