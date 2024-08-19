
#! This script is used to evaluate the performance of the planning. It uses the following metrics: 1. Planning time 2. Path length 3. Success rate

import os
import igl 
import numpy as np
import torch 
import time
import sys 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from main import Model 
from data_processing import utility



def save_stats_to_file(time_array, lengths_array, success_rate, filename):
    # Calculate mean and standard deviation for both arrays
    time_mean = np.mean(time_array)
    time_std = np.std(time_array)
    lengths_mean = np.mean(lengths_array)
    lengths_std = np.std(lengths_array)


    np.savez(filename, times=time_array, lengths=lengths_array, sc = success_rate)

    print(f"saved path eval data in {filename}")

def our_method_eval(experiment_folder, meshpath, modelpath, start_goal_points, vertices, faces):
    '''
    start_goal_points: a numpy array of shape (N, 2, 3)
    '''
    # #! load configs
    # config = utility.load_config(config_path)
    # name = config['paths']['name']
    # data_folder = os.path.join(config['paths'], name)
    # meshpath = os.path.join(data_folder, name + ".obj")

    #! load model
    model = Model(experiment_folder, config_path, device='cuda:0')
    model.load(modelpath)

    # #! load mesh
    # v, f = igl.read_triangle_mesh(meshpath)
    # vertices=v
    # faces=f
    # triangles = vertices[f]

    # start_goal_file = os.path.join(root_path, "valid_start_goal.npz")
    # data = np.load(start_goal_file)
    # start_points = data['start_points']
    # end_points = data['end_points']
    eval_times = []
    success_lens = []
    fail_trajs = []
    success_trajs = []
    trajectories = []
    for i in range(len(start_goal_points)):
        src = start_goal_points[i][0]
        tar = start_goal_points[i][1]
        start_time = time.time()
        cur_trajectory = model.predict_trajectory(src, tar, step_size=0.05, tol=0.05)
        end_time = time.time()
        # trajectories.append(cur_trajectory)
        
        if utility.check_collision(cur_trajectory, vertices, faces):
            lengths = utility.calculate_trajectory_lengths(cur_trajectory[None,])
            success_lens.append(lengths[0])
            success_trajs.append(cur_trajectory)
            eval_times.append(end_time - start_time)
        else:
            fail_trajs.append(cur_trajectory)

    success_rate = len(success_lens) / len(start_goal_points)
    eval_times = np.array(eval_times)
    success_lens = np.array(success_lens)

    outputfile = os.path.join(experiment_folder, "eval_traj_results.npz")
    save_stats_to_file(time_array=eval_times, lengths_array=success_lens, success_rate=success_rate, filename=outputfile)
    print("our fail trajs lens:", len(fail_trajs))
    print("success trajs lens:", len(success_lens))
    # return trajectories, fail_trajs
    #! vis
    if False:
        import trimesh 
        mesh = trimesh.load_mesh(meshpath)
        M = 5 #15
        trajs = success_trajs
        trajs = fail_trajs
        r = np.random.choice(len(trajs), M)
        for i in range(M):
            scene = trimesh.Scene(mesh)
            pc = trimesh.PointCloud(trajs[r[i]])
            scene.add_geometry([pc])
            scene.show()
        print()

if __name__ == "__main__":
    #! find the best model
    for dir in os.listdir("Experiments"):
        experiment_folder = os.path.join("Experiments", dir)
        config_file = [file for file in os.listdir(experiment_folder) if file.endswith(".yaml")][0]
        modelfiles = [file for file in os.listdir(experiment_folder) if file.endswith(".pt")]
        modelfiles.sort()
        best_model = modelfiles[-1]
        modelpath = os.path.join(experiment_folder, best_model)
        print(f"Best model: {best_model}")

        #! load config and query points
        config_path = os.path.join(experiment_folder, config_file)
        config = utility.load_config(config_path)
        name = config['paths']['name']
        data_folder = os.path.join(config['paths']['folder'], name)
        meshpath = os.path.join(data_folder, name + ".obj")

        #! load mesh
        v, f = igl.read_triangle_mesh(meshpath)
        t_obs = v[f]

        start_goal_points = utility.sample_points_inside_mesh(v, f, num_points=400).reshape(-1, 2, 3)

        our_method_eval(experiment_folder, meshpath, modelpath, start_goal_points, v, f)