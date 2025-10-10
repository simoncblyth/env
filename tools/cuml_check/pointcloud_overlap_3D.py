"""
pointcloud_overlap_3D.py
"""

import numpy as np
import pyvista as pv
from cuml.neighbors import NearestNeighbors
import cudf

# Parameters for two spheres
center1 = np.array([0.0, 0.0, 0.0])  # Center of first sphere
radius1 = 1.0  # Radius of first sphere
center2 = np.array([1.5, 0.0, 0.0])  # Center of second sphere
radius2 = 0.8  # Radius of second sphere
n_points = 10000  # Points per sphere (for denser sampling in 3D)
penetration_threshold = 0.0  # Signed distance < 0 indicates penetration

# Function to generate uniformly distributed points on a sphere
def generate_sphere_points(center, radius, n_points):
    u = np.random.rand(n_points)
    v = np.random.rand(n_points)
    theta = 2 * np.pi * u
    phi = np.arccos(2 * v - 1)
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)
    points = np.stack([x, y, z], axis=1) + center
    return points

# Generate points on two spheres
points1 = generate_sphere_points(center1, radius1, n_points)
points2 = generate_sphere_points(center2, radius2, n_points)

# Compute outward radial normals
normals1 = (points1 - center1) / radius1  # Unit vectors pointing outward
normals2 = (points2 - center2) / radius2  # Unit vectors pointing outward

# Convert to cuDF for cuML
points1_cudf = cudf.DataFrame(points1, columns=['x', 'y', 'z'])
points2_cudf = cudf.DataFrame(points2, columns=['x', 'y', 'z'])

# Bidirectional penetration detection

# Direction 1: Check if points2 penetrate into sphere1
nn_model = NearestNeighbors(n_neighbors=1, metric='euclidean')
nn_model.fit(points1_cudf)  # Fit on sphere1

distances21, indices21 = nn_model.kneighbors(points2_cudf)  # Query with sphere2
distances21 = distances21.to_numpy().flatten()
indices21 = indices21.to_numpy().flatten()
vectors21 = points2 - points1[indices21]
signed_distances21 = np.sum(vectors21 * normals1[indices21], axis=1)
penetration_mask21 = signed_distances21 < penetration_threshold
penetration_points2_into1 = points2[penetration_mask21]
penetration_corresponding1 = points1[indices21[penetration_mask21]]
penetration_signed21 = signed_distances21[penetration_mask21]

# Direction 2: Check if points1 penetrate into sphere2
nn_model.fit(points2_cudf)  # Fit on sphere2
distances12, indices12 = nn_model.kneighbors(points1_cudf)  # Query with sphere1
distances12 = distances12.to_numpy().flatten()
indices12 = indices12.to_numpy().flatten()
vectors12 = points1 - points2[indices12]
signed_distances12 = np.sum(vectors12 * normals2[indices12], axis=1)
penetration_mask12 = signed_distances12 < penetration_threshold
penetration_points1_into2 = points1[penetration_mask12]
penetration_corresponding2 = points2[indices12[penetration_mask12]]
penetration_signed12 = signed_distances12[penetration_mask12]

# Visualize with PyVista
plotter = pv.Plotter()

# Add point clouds
cloud1 = pv.PolyData(points1)
plotter.add_points(cloud1, color='blue', point_size=5, label='Sphere 1', opacity=0.5)

cloud2 = pv.PolyData(points2)
plotter.add_points(cloud2, color='green', point_size=5, label='Sphere 2', opacity=0.5)

# Add normal arrows (glyphs)
# For sphere1 normals
cloud1['normals'] = normals1
glyph1 = cloud1.glyph(orient='normals', scale=False, factor=0.1)  # factor controls arrow length
plotter.add_mesh(glyph1, color='blue', opacity=0.3)

# For sphere2 normals
cloud2['normals'] = normals2
glyph2 = cloud2.glyph(orient='normals', scale=False, factor=0.1)
plotter.add_mesh(glyph2, color='green', opacity=0.3)

# Highlight penetrating points (combine both directions)
if len(penetration_points2_into1) > 0:
    plotter.add_points(penetration_points2_into1, color='red', point_size=10, label='Penetrations (into Sphere 1)')
    plotter.add_points(penetration_corresponding1, color='red', point_size=10)
if len(penetration_points1_into2) > 0:
    plotter.add_points(penetration_points1_into2, color='red', point_size=10, label='Penetrations (into Sphere 2)')
    plotter.add_points(penetration_corresponding2, color='red', point_size=10)

# Add legend and show
plotter.add_legend()
plotter.show()

# Print penetration information
print(f"Number of points from Sphere 2 penetrating into Sphere 1: {len(penetration_points2_into1)}")
if len(penetration_points2_into1) > 0:
    print("Example pairs (Sphere 2 point -> Sphere 1 point):")
    for i in range(min(5, len(penetration_points2_into1))):
        print(f"Sphere 2: {penetration_points2_into1[i]}, Sphere 1: {penetration_corresponding1[i]}, "
              f"Signed Distance: {penetration_signed21[i]:.4f}")

print(f"Number of points from Sphere 1 penetrating into Sphere 2: {len(penetration_points1_into2)}")
if len(penetration_points1_into2) > 0:
    print("Example pairs (Sphere 1 point -> Sphere 2 point):")
    for i in range(min(5, len(penetration_points1_into2))):
        print(f"Sphere 1: {penetration_points1_into2[i]}, Sphere 2: {penetration_corresponding2[i]}, "
              f"Signed Distance: {penetration_signed12[i]:.4f}")



