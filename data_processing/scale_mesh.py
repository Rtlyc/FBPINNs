import trimesh
import numpy as np

# Load your mesh
meshpath = 'data/auburn.obj'
mesh = trimesh.load(meshpath)

# Step 1: Translate the mesh so that its center is at the origin
center = mesh.centroid
mesh.apply_translation(-center)

# Step 2: Scale the mesh to fit within [-3, 3]
# Calculate the current bounding box size
bbox = mesh.bounds
size = bbox[1] - bbox[0]  # Size in each dimension

# Find the maximum extent to scale uniformly
max_extent = np.max(size)

# Calculate the scaling factor to fit within [-3, 3]
scale_factor = 20.0 / max_extent

# Apply scaling
mesh.apply_scale(scale_factor)

# Now, the mesh is centered at the origin and scaled to fit within [-3, 3].

# Save the transformed mesh if needed
mesh.export('scaled_translated_mesh.obj')
