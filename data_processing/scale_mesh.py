import trimesh
import numpy as np
import os
import yaml

def scale_mesh(meshpath, targetpath, scale_length=10.0):
    # Load your mesh
    mesh = trimesh.load(meshpath)

    # # Step 1: Translate the mesh so that its center is at the origin
    # center = mesh.centroid
    # mesh.apply_translation(-center)

    # Step 2: Scale the mesh to fit within [-3, 3]
    # Calculate the current bounding box size
    bbox = mesh.bounds
    size = bbox[1] - bbox[0]  # Size in each dimension

    # Find the maximum extent to scale uniformly
    max_extent = np.max(size)

    # Calculate the scaling factor to fit within [-3, 3]
    scale_factor = scale_length / max_extent

    # Apply scaling
    mesh.apply_scale(scale_factor)

    center = mesh.bounds.mean(axis=0)
    mesh.apply_translation(-center)

    # Save the transformed mesh if needed
    mesh.export(targetpath)


if __name__ == "__main__":
    # mesh_folder = "data/gibson_mesh"
    config_folder = "configs/gibson_configs"
    config_folder = "configs/gibson_all_configs"
    for config_file in os.listdir(config_folder):
        if config_file.endswith(".yaml"):
            with open(os.path.join(config_folder, config_file), 'r') as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
            name = config['paths']['name']
            raw_meshpath = config['paths']['raw_meshpath']
            folder = config['paths']['folder']
            folder = os.path.join(folder, name)
            if not os.path.exists(folder):
                os.makedirs(folder)
            target_meshpath = os.path.join(folder, name + ".obj")
            scale_mesh(raw_meshpath, target_meshpath)
            print(f"Processed: {name}, saved to {target_meshpath}")

            if False:
                scene = trimesh.Scene()
                mesh = trimesh.load_mesh(target_meshpath)
                scene.add_geometry(mesh)
                # Define line segments for X (red), Y (green), and Z (blue) axes
                axis_length = 5.0
                x_axis = trimesh.load_path(np.array([[0, 0, 0], [axis_length, 0, 0]]))
                y_axis = trimesh.load_path(np.array([[0, 0, 0], [0, axis_length, 0]]))
                z_axis = trimesh.load_path(np.array([[0, 0, 0], [0, 0, axis_length]]))
                x_axis.colors = [[255, 0, 0, 255]]
                y_axis.colors = [[0, 255, 0, 255]]
                z_axis.colors = [[0, 0, 255, 255]]
                scene.add_geometry([ x_axis, y_axis, z_axis])
                scene.show()


    # # Iterate over all mesh files in the folder
    # for mesh_file in os.listdir(mesh_folder):
    #     if mesh_file.endswith(".obj"):
    #         meshpath = os.path.join(mesh_folder, mesh_file)
    #         targetpath = os.path.join(mesh_target_folder, mesh_file)
    #         scale_mesh(meshpath, targetpath)
    #         print(f"Processed: {mesh_file}")
