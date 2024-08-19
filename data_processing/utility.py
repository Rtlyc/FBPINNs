import numpy as np
import igl 
import torch 
import yaml
import os
import datetime

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def save_config(config, directory="configs", filename=None):
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    if filename is None:
        # Create a timestamped filename
        filename = datetime.now().strftime("%Y%m%d_%H%M%S") + "_config.yaml"
    
    filepath = os.path.join(directory, filename)
    
    with open(filepath, 'w') as f:
        yaml.dump(config, f)
    
    print(f"Configuration saved to {filepath}")

def unsigned_distance_without_bvh(vertices, faces, query_points):
    # Assuming your tensors are called triangles and query_points
    # triangles_np = triangles
    query_points_np = query_points

    # Flatten and get unique vertices
    # vertices, inverse_indices = np.unique(triangles_np, axis=0, return_inverse=True)

    # # Convert back the inverse indices to get faces
    # faces = inverse_indices.reshape(-1, 3)



    # Compute the squared distance (Note: to get the actual distance, take the sqrt of the results)
    squared_d, closest_faces, closest_points = igl.point_mesh_squared_distance(query_points_np, vertices, faces)

    # distances would be the sqrt of squared_d
    unsigned_distance = np.sqrt(squared_d)

    return unsigned_distance


def sample_points_inside_mesh(vertices, faces, num_points):
    """
    Sample points inside a mesh using rejection sampling.
    """
    # Read the mesh using libigl
    # v, f = igl.read_triangle_mesh(meshpath)
    
    # Compute the axis-aligned bounding box (AABB) of the mesh
    min_corner = vertices.min(axis=0)
    max_corner = vertices.max(axis=0)

    points = []
    while len(points) < num_points:
        # Generate random points within the bounding box
        grid_points = np.random.uniform(min_corner, max_corner, (num_points, 3))
        
        # Compute winding numbers for the generated points
        winding_numbers = igl.winding_number(vertices, faces, grid_points)
        
        # Filter points that are inside the mesh based on winding numbers
        points_inside = grid_points[np.abs(winding_numbers) > 0.1]
        
        # Append the filtered points to the list
        points.extend(points_inside)
    
    # Return the desired number of sampled points
    return np.array(points)[:num_points]

def check_collision(query_points, vertices, faces, scale=1):
    query_points /= scale 
    query_points = torch.tensor(query_points, dtype=torch.float32, device='cuda')
    
    #! interpolate points
    new_query_points = []
    interpolation_steps = 10
    for i in range(len(query_points) - 1):
        start_point = query_points[i]
        end_point = query_points[i+1]
        for t in range(interpolation_steps):
            alpha = t/interpolation_steps
            interpolated_point = (1-alpha)*start_point + alpha*end_point
            new_query_points.append(interpolated_point)
    new_query_points = torch.vstack(new_query_points)
    query_points = new_query_points
    
    query_points = query_points
    #print(query_points.shape)
    # bvh = bvh_distance_queries.BVH()
    # torch.cuda.synchronize()
    # torch.cuda.synchronize()
    # distances, closest_points, closest_faces, closest_bcs= bvh(triangles, query_points)
    # torch.cuda.synchronize()
    # #unsigned_distance = abs()
    # #print(distances.shape)
    # unsigned_distance = torch.sqrt(distances).squeeze()
    unsigned_distance = unsigned_distance_without_bvh(vertices, faces, query_points.cpu().numpy())
    # unsigned_distance = unsigned_distance.detach().cpu().numpy()
    
    iter = 0 #! used for trajectory for NTFields
    if np.min(unsigned_distance)<=0.001 or iter>500:
        return False
    else:
        return True

def calculate_trajectory_lengths(trajectories):
    # This function calculates the length of each trajectory.
    # trajectories: a numpy array of shape (N, 64, 3)
    nt = trajectories.shape[0]
    lengths = np.zeros(nt)

    for i in range(nt):
        for j in range(1, trajectories.shape[1]):
            lengths[i] += np.linalg.norm(trajectories[i, j, :] - trajectories[i, j-1, :])
    
    return lengths

if __name__ == "__main__":
    v, f = igl.read_triangle_mesh("data/auburn.obj")
    points = sample_points_inside_mesh(v, f, 1000)
    print(points.shape)