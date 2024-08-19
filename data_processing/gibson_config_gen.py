import os 
import yaml 
import utility

def generate_gibson_config(gibson_folder, target_folder, data_folder, example_config, filenum=2):
    '''
    Generate Gibson config files for the meshes in the gibson_folder.
        gibson_folder: str, the folder containing the meshes.
        target_folder: str, the folder to save the generated config files.
        data_folder: str, the folder to save the data.
        example_config: str, the path to the example config file.
        filenum: int, the number of files to generate config files
    '''
    config = utility.load_config(example_config)
    for dir in os.listdir(gibson_folder)[:filenum]:
        temp_config = config.copy()
        name = dir 
        raw_meshpath = os.path.join(gibson_folder, dir, "mesh_z_up.obj")
        temp_config['paths']['name'] = name
        temp_config['paths']['raw_meshpath'] = raw_meshpath
        temp_config['paths']['folder'] = data_folder
        utility.save_config(temp_config, target_folder, name + ".yaml")



if __name__ == "__main__":
    gibson_folder = "/home/exx/Documents/gibson_large_plus_mesh/gibson_fullplus"
    target_folder = "configs/gibson_all_configs"
    data_folder = "data/gibson_all"
    example_config = "configs/example_gibson.yaml"
    generate_gibson_config(gibson_folder, target_folder, data_folder, example_config, filenum=100)
    print("Gibson config files generated.")

