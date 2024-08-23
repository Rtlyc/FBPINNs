from mesh_sample import viz 
from utility import load_config
import os
import numpy as np

if __name__ == '__main__':
    # config_path = "configs/final_gibson_config/Almena.yaml"
    config_folder = "configs/final_gibson_config"
    for config_file in os.listdir(config_folder)[0:1]:
        if config_file.endswith(".yaml"):
            config_path = os.path.join(config_folder, config_file)
            config = load_config(config_path)
            data_folder = config['paths']['folder']
            name = config['paths']['name']
            data_folder = os.path.join(data_folder, name)
            print("visualizing", name)
            pointspath = os.path.join(data_folder, f"{name}_points_0.npy")
            speedspath = os.path.join(data_folder, f"{name}_speeds_0.npy")
            meshpath = os.path.join(data_folder, f"{name}.obj")
            points = np.load(pointspath)
            speeds = np.load(speedspath)

            viz(points, speeds, meshpath=meshpath, viz_start=False)