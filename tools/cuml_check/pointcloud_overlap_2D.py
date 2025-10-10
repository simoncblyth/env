"""
pointcloud_overlap_2D.py

"""

import numpy as np
import matplotlib.pyplot as plt
from cuml.neighbors import NearestNeighbors
import cudf

# Parameters for two circles
center1 = np.array([0.0, 0.0])  # Center of first circle
radius1 = 1.0  # Radius of first circle
center2 = np.array([1.5, 0.0])  # Center of second circle
radius2 = 0.8  # Radius of second circle
n_points = 100  # Points per circle
penetration_threshold = 0.0  # Signed distance < 0 indicates penetration

# Generate points on two circles
theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
points1 = center1 + radius1 * np.vstack([np.cos(theta), np.sin(theta)]).T
points2 = center2 + radius2 * np.vstack([np.cos(theta), np.sin(theta)]).T

# Compute outward radial normals
normals1 = (points1 - center1) / radius1  # Unit vectors pointing outward
normals2 = (points2 - center2) / radius2  # Unit vectors pointing outward

# Convert to cuDF for cuML
points1_cudf = cudf.DataFrame(points1, columns=['x', 'y'])
points2_cudf = cudf.DataFrame(points2, columns=['x', 'y'])

# Use cuML NearestNeighbors to find closest points
nn_model = NearestNeighbors(n_neighbors=1, metric='euclidean')
nn_model.fit(points1_cudf)  # Fit on first point cloud
distances, indices = nn_model.kneighbors(points2_cudf)  # Query with second point cloud

# Convert results to numpy
distances = distances.to_numpy().flatten()
indices = indices.to_numpy().flatten()

# Vectorized signed distance computation
# Vector from p2 to nearest p1: p2 - p1[indices]
vectors = points2 - points1[indices]
# Signed distance: (p2 - p1) dot n1 for each nearest neighbor pair
signed_distances = np.sum(vectors * normals1[indices], axis=1)

# Identify penetrating points (signed distance < threshold)
penetration_mask = signed_distances < penetration_threshold
penetration_points2 = points2[penetration_mask]
penetration_points1 = points1[indices[penetration_mask]]
penetration_signed_distances = signed_distances[penetration_mask]

# Visualize point clouds, normals, and penetrations
plt.figure(figsize=(8, 8))

# Plot points
plt.scatter(points1[:, 0], points1[:, 1], c='blue', s=50, alpha=0.5, label='Circle 1')
plt.scatter(points2[:, 0], points2[:, 1], c='green', s=50, alpha=0.5, label='Circle 2')

# Plot normal arrows
arrow_scale = 0.2  # Length of normal arrows
plt.quiver(
    points1[:, 0], points1[:, 1], normals1[:, 0], normals1[:, 1],
    color='blue', scale=1/arrow_scale, width=0.002, alpha=0.5
)
plt.quiver(
    points2[:, 0], points2[:, 1], normals2[:, 0], normals2[:, 1],
    color='green', scale=1/arrow_scale, width=0.002, alpha=0.5
)

# Highlight penetrating points
if len(penetration_points2) > 0:
    plt.scatter(penetration_points2[:, 0], penetration_points2[:, 1], 
                c='red', s=100, marker='x', label='Penetrations (Circle 2)')
    plt.scatter(penetration_points1[:, 0], penetration_points1[:, 1], 
                c='red', s=100, marker='x')

# Add circle outlines for clarity
circle1 = plt.Circle(center1, radius1, color='blue', fill=False, linestyle='--', alpha=0.3)
circle2 = plt.Circle(center2, radius2, color='green', fill=False, linestyle='--', alpha=0.3)
plt.gca().add_patch(circle1)
plt.gca().add_patch(circle2)

plt.title('Two Circular Point Clouds with Outward Normals and Penetrations')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.legend()
plt.axis('equal')  # Equal scaling for x and y axes
plt.show()

# Print penetration information
print(f"Number of penetrating points: {len(penetration_points2)}")
if len(penetration_points2) > 0:
    print("Example penetrating pairs (Circle 2 point -> Circle 1 point):")
    for i in range(min(5, len(penetration_points2))):
        print(f"Circle 2: {penetration_points2[i]}, Circle 1: {penetration_points1[i]}, "
              f"Signed Distance: {penetration_signed_distances[i]:.4f}")






