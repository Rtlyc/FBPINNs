import numpy as np
from torch.utils.data import Dataset 
import torch 
from scipy.spatial.transform import Rotation as R 
import os, cv2, json
from torchvision import transforms
# from dataprocessing import transform, data_pc_validate
import matplotlib.pyplot as plt
import igl
from mesh_sample import viz

PATH = "/home/exx/Documents/FBPINNs/b1/haas_first/hass_first_floor"

def collect_files(path=PATH,file_start = 0, file_end=100):
    files_in_range = []
    for file in os.listdir(path):
        # split the file
        file_id = file.split("_")[0]
        if file_start <= int(file_id) <= file_end:
            files_in_range.append((int(file_id), os.path.join(PATH, file)))
    files_in_range.sort()
    return files_in_range

def transform_pc_to_local(pc, rotation_matrix):
    pc = np.einsum("ij,kj->ik", pc, rotation_matrix)
    return pc 

def get_lidar_data(path="/home/exx/Documents/FBPINNs/b1/haas_first/hass_first_floor"):
    tgt_files = collect_files(path)
    pcl_data = []
    for dir_id, dir in tgt_files[::1]:
        # if dir_id % 2 == 0:
            # continue
        json_path = os.path.join(dir, 'metadata.json')
        
        metadata = {}
        with open(json_path) as f:
            metadata = json.load(f)
        pcl_path = metadata['pc_loc']
        pcl_path = os.path.join(dir, "pcl.npy")
        pcl = np.load(pcl_path)
        
        position_data = metadata['pose']['position']
        orientation_data = metadata['pose']['orientation']
        r = R.from_quat([orientation_data['x'], 
                        orientation_data['y'], 
                        orientation_data['z'], 
                        orientation_data['w']])
        # print(r.as_euler("XYZ", degrees=True))
        # rotation_matrix = -r.as_matrix()
        head_angle = r.as_euler("XYZ", degrees=False)[2]
        print(f'{position_data["x"]:.2f}', f'{position_data["y"]:.2f}', f'{head_angle:.2f}')

        rot_mat = np.array([[np.cos(head_angle),-np.sin(head_angle), 0],
                            [np.sin(head_angle),np.cos(head_angle), 0],
                            [0,0,1]])
        translation_data = np.array([position_data['x'], position_data['y'], 0])
        pc_local = transform_pc_to_local(pcl, rot_mat)
        pcl_data.append((pc_local,translation_data))
    pcl_data = np.array(pcl_data, dtype=object)
    return pcl_data[:, 0], pcl_data[:, 1]

def get_lidar_data2(path="/home/exx/Documents/FBPINNs/b1/haas_first/hass_first_floor"):
    lidar_path = os.path.join(path, "lidar.npy")
    positions_path = os.path.join(path, "positions.npy")
    lidar = np.load(lidar_path, allow_pickle=True)
    positions = np.load(positions_path)
    return lidar, positions


class LidarDataset(Dataset):
    def __init__(
        self,
        root_dir,
        config=None,   
    ):
        self.root_dir = root_dir
        self.config_file = config
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        with open(self.config_file) as f:
            configs = json.load(f)

        depth_scale = configs["dataset"]["depth_scale"]
        inv_depth_scale = 1. / depth_scale
        self.min_depth = configs["sample"]["depth_range"][0]
        self.max_depth = configs["sample"]["depth_range"][1]
        self.n_rays = configs["sample"]["n_rays"]
        self.n_rays = 5000 #5000
        self.n_strat_samples = configs["sample"]["n_strat_samples"]
        self.n_surf_samples = configs["sample"]["n_surf_samples"]
        self.dist_behind_surf = configs["sample"]["dist_behind_surf"]  


        self.up_vec = np.array([0., 1., 0.])
        # self.dirs_C = transform.ray_dirs_C(1,self.H,self.W,self.fx,self.fy,self.cx,self.cy,self.device,depth_type="z")
        #TODO: add self.dirs_W
        self.lidar_data, self.Ts = get_lidar_data2(self.root_dir)
        # self.frame_data = np.array(self.frame_data, dtype=object)

        # self.depthfiles, self.posefiles, self.rgbfiles = data_pc_validate.get_filenames_for_tbot(self.root_dir)

        # self.Ts = []
        # self.skip = 5
        # for idx in range(0, len(self.posefiles), self.skip):
        #     pose = np.load(self.posefiles[idx], allow_pickle=True)
        #     pose = data_pc_validate.convert_transform_odom_to_cam(pose)
        #     temp = pose.copy()
        #     pose_pred = data_pc_validate.convert_cam_xyz_to_odom_xyz(temp)
        #     pose[0, 3] -= 3
        #     self.Ts.append(pose)
        # self.Ts = np.array(self.Ts)

        print("dataset length: ", len(self))

    def __len__(self):
        return len(self.Ts)
    
    def __getitem__(self, idx):
        #! provide depth, depth_dirs, T
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()
        
        # s = f"{idx:06}"
        # depth_file = os.path.join(self.root_dir,"depth" + s + ".npy")
        
        # #TODO: change to my own depth file, this depth is local
        # depth = np.load(depth_file)
        depths = self.lidar_data[idx]
        T = self.Ts[idx]
        depth_norms = np.linalg.norm(depths, axis=-1)
        depth_dirs = depths/depth_norms[:, None]
        sample = {"depth": depth_norms, "T": T, "depth_dirs": depth_dirs}

        # T = None 
        # if self.Ts is not None:
        #     T = self.Ts[idx]

        # if self.use_lidar:
        if False:
            depth_dirs, depth = depth, np.linalg.norm(depth, axis=-1)
            rotation_matrix_90_z = np.array([
                [0, 1, 0],
                [-1, 0, 0],
                [0, 0, 1]
            ])
            depth_dirs = depth_dirs.dot(rotation_matrix_90_z.T)/depth[:, None]
             
            
        
        # sample = {"depth": depth, "T": T}

        # # if self.use_lidar:
        # sample["depth_dirs"] = depth_dirs



        return sample
    
    def get_speeds(self, idx_list, minimum=0.07, maximum=0.3, num=10000, is_gt_speed=False):
        all_points = []
        all_bounds = []
        device = self.device
        for idx in idx_list:
            sub_dataset = self[idx]
            depth_np = sub_dataset["depth"][None, ...]
            T_np = sub_dataset["T"][None, ...]

            depth = torch.from_numpy(depth_np).float().to(device)
            T = torch.from_numpy(T_np).float().to(device)
            # pc = transform.pointcloud_from_depth_torch(depth[0], self.fx, self.fy, self.cx, self.cy)

            depth_dirs_np = sub_dataset["depth_dirs"][None, ...]
            depth_dirs = torch.from_numpy(depth_dirs_np).float().to(device)
            sample_pts = sample_lidar_points(depth, T, self.n_rays, depth_dirs, self.dist_behind_surf, self.n_strat_samples, self.n_surf_samples, self.min_depth, self.max_depth, device=device)

            bound = bounds_pc(sample_pts["pc"], sample_pts["z_vals"],
                              sample_pts["surf_pc"], sample_pts["depth_sample"])
            all_points.append(sample_pts["pc"])
            all_bounds.append(bound)
        
        pc = torch.cat(all_points, dim=0)
        bounds = torch.cat(all_bounds, dim=0)
        pc = pc.view(-1, 3)
        bounds = bounds.view(-1, 1)

        if is_gt_speed: #ground truth
            bounds = self.get_gt_bounds("datasets/igib-seqs/Beechwood_0_int_scene_mesh.obj", pc)
        
        speeds = torch.clip(bounds, minimum, maximum)/maximum

        valid_indices = torch.where((speeds <= 1) & (speeds > 0))[0] 
        if num <= len(valid_indices):
            # Select without replacement if num is less than or equal to the size of valid_indices
            start_indices = valid_indices[torch.randperm(len(valid_indices))[:num]]
            end_indices = valid_indices[torch.randperm(len(valid_indices))[:num]]
        else:
            # Select with replacement if num is greater than the size of valid_indices
            rand_indices = torch.randint(0, len(valid_indices), (num,))
            start_indices = valid_indices[rand_indices]

            rand_indices = torch.randint(0, len(valid_indices), (num,))
            end_indices = valid_indices[rand_indices]

        end_indices = torch.randint(0, pc.shape[0], (num,))
        x0 = pc[start_indices]
        x1 = pc[end_indices]
        x = torch.cat((x0, x1), dim=1)
        y = torch.cat((speeds[start_indices], speeds[end_indices]), dim=1)
        z = torch.cat((bounds[start_indices], bounds[end_indices]), dim=1)

        return x,y,z # points, speeds, bounds


def sample_lidar_points(depth, T_WC, n_rays, dirs_W, dist_behind_surf, n_strat_samples, n_surf_samples, min_depth, max_depth, device='cpu'):
    torch.manual_seed(0)
    indices = torch.randint(0, depth.shape[1], (n_rays,), device=device)
    depth_sample = depth[0][indices] #? could cause problem, check dimension
    mask_valid_depth = depth_sample > min_depth
    depth_sample = depth_sample[mask_valid_depth]
    indices = indices[mask_valid_depth]
    dirs_W_sample = dirs_W[:, indices].view(-1, 3)

    max_sample_depth = torch.min(depth_sample + dist_behind_surf, torch.tensor(max_depth))
    # max_sample_depth = depth_sample

    if True:
        pc, z_vals, surf_pc = sample_along_lidar_rays(T_WC, min_depth, max_sample_depth, n_strat_samples, n_surf_samples, dirs_W_sample, depth_sample)
    else:
        #TODO: need a new sampling strategy, output surf_pc and pc
        numsamples = 5000 
        dim = 3
        OutsideSize = numsamples + 2
        WholeSize = 0
        origin = T_WC[0, :3, 3]
        origins = T_WC[:, :3, 3]
        surf_pc = origins + (dirs_W_sample * depth_sample[:, None])
        while OutsideSize > 0:
            P  = origin + torch.rand((15*numsamples,dim),dtype=torch.float32, device='cuda')-0.5 # random start point
            dP = torch.rand((15*numsamples,dim),dtype=torch.float32, device='cuda')-0.5 # random direction
            rL = (torch.rand((15*numsamples,1),dtype=torch.float32, device='cuda'))*(max_depth) # random length
            nP = P + torch.nn.functional.normalize(dP,dim=1)*rL

            # need our own PointsInside
            # PointsInside = torch.all((nP <= 0.5),dim=1) & torch.all((nP >= -0.5),dim=1)
            print(depth_sample)
            bounds = bounds_pc(P, surf_pc, depth_sample)

            x0 = P[PointsInside, :]
            x1 = nP[PointsInside, :]
            if (x0.shape[0]<=1):
                continue 

            # obs_distance0 = bounds_pc(x0, surf_pc)
            # where_d = (obs_distance0 > minimum) & (obs_distance0 < maximum)
            OutsideSize = OutsideSize - x0.shape[0]
            WholeSize = WholeSize + x0.shape[0]

            if WholeSize > numsamples:
                break


    sample_pts = {
        "depth_batch": depth, 
        "pc": pc,
        "z_vals": z_vals,
        "surf_pc": surf_pc,
        "indices": indices,
        "dirs_W_sample": dirs_W_sample,
        "depth_sample": depth_sample,
        "T_WC": T_WC,
    }

    return sample_pts 


def sample_along_lidar_rays(T_WC, min_depth, max_depth, n_stratified_samples, n_surf_samples, dirs_W, gt_depth):
    #rays in world frame    
    # origins = T_WC[:, :3, 3]
    origins = T_WC
    dirs_W = dirs_W.view(-1, 3)
    n_rays = dirs_W.shape[0]

    #stratified samples along rays
    z_vals = stratified_sample(min_depth, max_depth, n_rays, T_WC.device, n_stratified_samples)


    # if gt_depth is given, first sample at surface then around surface
    if gt_depth is not None and n_surf_samples > 0:
        surface_z_vals = gt_depth 
        offsets = torch.normal(torch.zeros(gt_depth.shape[0], n_surf_samples-1), 0.1).to(z_vals.device)
        near_surf_z_vals = gt_depth[:,None] + offsets
        if not isinstance(min_depth, torch.Tensor):
            min_depth = torch.full(near_surf_z_vals.shape, min_depth).to(z_vals.device)[...,0]
        near_surf_z_vals = torch.clamp(near_surf_z_vals, min_depth[:,None], max_depth[:,None])

        z_vals = torch.cat((near_surf_z_vals, z_vals), dim=1)

    # point cloud of 3d sample locations
    pc = origins[:, None, :] + (dirs_W[:, None, :] * z_vals[:, :, None])
    
    # surf_pc filter out points on the ground
    surf_pc = origins + (dirs_W * gt_depth[:, None])
    # isNotGround = surf_pc[:, 2] > 0.01
    isNotGround = surf_pc[:, 2] > -0.11
    # surf_pc = surf_pc[isNotGround]

    return pc, z_vals, surf_pc

def stratified_sample(
    min_depth,
    max_depth,
    n_rays,
    device,
    n_stratified_samples,
    bin_length=None,
):
    """
    Random samples between min and max depth
    One sample from within each bin.

    If n_stratified_samples is passed then use fixed number of bins,
    else if bin_length is passed use fixed bin size.
    """
    if n_stratified_samples is not None:  # fixed number of bins
        n_bins = n_stratified_samples
        if isinstance(max_depth, torch.Tensor):
            sample_range = (max_depth - min_depth)[:, None]
            bin_limits = torch.linspace(
                0, 1, n_bins + 1,
                device=device)[None, :]
            bin_limits = bin_limits.repeat(n_rays, 1) * sample_range
            if isinstance(min_depth, torch.Tensor):
                bin_limits = bin_limits + min_depth[:, None]
            else:
                bin_limits = bin_limits + min_depth
            bin_length = sample_range / (n_bins)
        else:
            bin_limits = torch.linspace(
                min_depth,
                max_depth,
                n_bins + 1,
                device=device,
            )[None, :]
            bin_length = (max_depth - min_depth) / (n_bins)

    elif bin_length is not None:  # fixed size of bins
        bin_limits = torch.arange(
            min_depth,
            max_depth,
            bin_length,
            device=device,
        )[None, :]
        n_bins = bin_limits.size(1) - 1

    increments = torch.rand(n_rays, n_bins, device=device) * bin_length
    # increments = 0.5 * torch.ones(n_rays, n_bins, device=device) * bin_length
    lower_limits = bin_limits[..., :-1]
    z_vals = lower_limits + increments

    return z_vals


def bounds_pc(pc, z_vals, surf_pc, depth_sample):
    # surf_pc = pc[:, 0]
    diff = pc[:, :, None] - surf_pc 
    dists = diff.norm(dim=-1)
    dists, closest_ixs = dists.min(axis=-1)
    behind_surf = z_vals > depth_sample[:, None]
    dists[behind_surf] *= -1
    #! set to 0 if behind surface
    # dists[behind_surf] = 0
    bounds = dists

    return bounds



if __name__ == "__main__":
    #! test the dataset
    root_dir = "/home/exx/Documents/FBPINNs/b1/haas_first/haas_test"
    root_dir = "/home/exx/Documents/FBPINNs/b1/haas_first/hass_first_floor"
    root_dir = '/home/exx/Documents/FBPINNs/b1/haas'
    config_file = "/home/exx/Documents/FBPINNs/configs/lidar.json"
    dataset = LidarDataset(root_dir, config_file)
    sample = dataset[0]
    print(sample["depth"].shape)
    print(sample["T"].shape)
    print(sample["depth_dirs"].shape)
    points, speeds, bounds = dataset.get_speeds(range(94), num=20000) 
    viz(points.cpu().numpy(), speeds.cpu().numpy())   








