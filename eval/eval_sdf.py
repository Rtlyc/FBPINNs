import numpy as np 
import torch 
import igl 
import os 
import sys 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main import Model
from data_processing.utility import sample_points_inside_mesh, unsigned_distance_without_bvh

import time
# from start_goal_rand_gen import generate_points
# import bvh_distance_queries
from torch import Tensor 
from torch.autograd import Variable 
import skimage




import trimesh
import matplotlib.pyplot as plt


def grid_sample_3d(image, optical):
    N, C, ID, IH, IW = image.shape
    _, D, H, W, _ = optical.shape

    ix = optical[..., 0]
    iy = optical[..., 1]
    iz = optical[..., 2]

    ix = ((ix + 0.5) ) * (IW - 1)
    iy = ((iy + 0.5) ) * (IH - 1)
    iz = ((iz + 0.5) ) * (ID - 1)
    with torch.no_grad():
        
        ix_tnw = torch.floor(ix)
        iy_tnw = torch.floor(iy)
        iz_tnw = torch.floor(iz)

        ix_tne = ix_tnw + 1
        iy_tne = iy_tnw
        iz_tne = iz_tnw

        ix_tsw = ix_tnw
        iy_tsw = iy_tnw + 1
        iz_tsw = iz_tnw

        ix_tse = ix_tnw + 1
        iy_tse = iy_tnw + 1
        iz_tse = iz_tnw

        ix_bnw = ix_tnw
        iy_bnw = iy_tnw
        iz_bnw = iz_tnw + 1

        ix_bne = ix_tnw + 1
        iy_bne = iy_tnw
        iz_bne = iz_tnw + 1

        ix_bsw = ix_tnw
        iy_bsw = iy_tnw + 1
        iz_bsw = iz_tnw + 1

        ix_bse = ix_tnw + 1
        iy_bse = iy_tnw + 1
        iz_bse = iz_tnw + 1

    tnw = (ix_bse - ix) * (iy_bse - iy) * (iz_bse - iz)
    tne = (ix - ix_bsw) * (iy_bsw - iy) * (iz_bsw - iz)
    tsw = (ix_bne - ix) * (iy - iy_bne) * (iz_bne - iz)
    tse = (ix - ix_bnw) * (iy - iy_bnw) * (iz_bnw - iz)
    bnw = (ix_tse - ix) * (iy_tse - iy) * (iz - iz_tse)
    bne = (ix - ix_tsw) * (iy_tsw - iy) * (iz - iz_tsw)
    bsw = (ix_tne - ix) * (iy - iy_tne) * (iz - iz_tne)
    bse = (ix - ix_tnw) * (iy - iy_tnw) * (iz - iz_tnw)


    with torch.no_grad():

        torch.clamp(ix_tnw, 0, IW - 1, out=ix_tnw)
        torch.clamp(iy_tnw, 0, IH - 1, out=iy_tnw)
        torch.clamp(iz_tnw, 0, ID - 1, out=iz_tnw)

        torch.clamp(ix_tne, 0, IW - 1, out=ix_tne)
        torch.clamp(iy_tne, 0, IH - 1, out=iy_tne)
        torch.clamp(iz_tne, 0, ID - 1, out=iz_tne)

        torch.clamp(ix_tsw, 0, IW - 1, out=ix_tsw)
        torch.clamp(iy_tsw, 0, IH - 1, out=iy_tsw)
        torch.clamp(iz_tsw, 0, ID - 1, out=iz_tsw)

        torch.clamp(ix_tse, 0, IW - 1, out=ix_tse)
        torch.clamp(iy_tse, 0, IH - 1, out=iy_tse)
        torch.clamp(iz_tse, 0, ID - 1, out=iz_tse)

        torch.clamp(ix_bnw, 0, IW - 1, out=ix_bnw)
        torch.clamp(iy_bnw, 0, IH - 1, out=iy_bnw)
        torch.clamp(iz_bnw, 0, ID - 1, out=iz_bnw)

        torch.clamp(ix_bne, 0, IW - 1, out=ix_bne)
        torch.clamp(iy_bne, 0, IH - 1, out=iy_bne)
        torch.clamp(iz_bne, 0, ID - 1, out=iz_bne)

        torch.clamp(ix_bsw, 0, IW - 1, out=ix_bsw)
        torch.clamp(iy_bsw, 0, IH - 1, out=iy_bsw)
        torch.clamp(iz_bsw, 0, ID - 1, out=iz_bsw)

        torch.clamp(ix_bse, 0, IW - 1, out=ix_bse)
        torch.clamp(iy_bse, 0, IH - 1, out=iy_bse)
        torch.clamp(iz_bse, 0, ID - 1, out=iz_bse)

    image = image.view(N, C, ID * IH * IW)

    tnw_val = torch.gather(image, 2, (iz_tnw * IW * IH + iy_tnw * IW + ix_tnw).long().view(N, 1, D * H * W).repeat(1, C, 1))
    tne_val = torch.gather(image, 2, (iz_tne * IW * IH + iy_tne * IW + ix_tne).long().view(N, 1, D * H * W).repeat(1, C, 1))
    tsw_val = torch.gather(image, 2, (iz_tsw * IW * IH + iy_tsw * IW + ix_tsw).long().view(N, 1, D * H * W).repeat(1, C, 1))
    tse_val = torch.gather(image, 2, (iz_tse * IW * IH + iy_tse * IW + ix_tse).long().view(N, 1, D * H * W).repeat(1, C, 1))
    bnw_val = torch.gather(image, 2, (iz_bnw * IW * IH + iy_bnw * IW + ix_bnw).long().view(N, 1, D * H * W).repeat(1, C, 1))
    bne_val = torch.gather(image, 2, (iz_bne * IW * IH + iy_bne * IW + ix_bne).long().view(N, 1, D * H * W).repeat(1, C, 1))
    bsw_val = torch.gather(image, 2, (iz_bsw * IW * IH + iy_bsw * IW + ix_bsw).long().view(N, 1, D * H * W).repeat(1, C, 1))
    bse_val = torch.gather(image, 2, (iz_bse * IW * IH + iy_bse * IW + ix_bse).long().view(N, 1, D * H * W).repeat(1, C, 1))

    out_val = ( tnw_val.view(N, C, D, H, W) * tnw.view(N, 1, D, H, W) +
                tne_val.view(N, C, D, H, W) * tne.view(N, 1, D, H, W) +
                tsw_val.view(N, C, D, H, W) * tsw.view(N, 1, D, H, W) +
                tse_val.view(N, C, D, H, W) * tse.view(N, 1, D, H, W) +
                bnw_val.view(N, C, D, H, W) * bnw.view(N, 1, D, H, W) +
                bne_val.view(N, C, D, H, W) * bne.view(N, 1, D, H, W) +
                bsw_val.view(N, C, D, H, W) * bsw.view(N, 1, D, H, W) +
                bse_val.view(N, C, D, H, W) * bse.view(N, 1, D, H, W))

    return out_val

def gt_tsdf(t_obs, query_points, minimum = 0.02, maximum = 0.2):
    unsigned_distance = unsigned_distance_without_bvh(t_obs, query_points)

    gt_tsdf = np.clip(unsigned_distance, minimum, maximum)/maximum
    return gt_tsdf

def our_tsdf(experiment_folder, modelpath, config_path, query_points):
    model = Model(experiment_folder, config_path, device='cuda:0')
    model.load(modelpath)

    allstart = np.zeros((len(query_points), 3))
    input_data = np.concatenate((allstart, query_points), axis=1)
    input_data_torch = torch.from_numpy(input_data).float().cuda()
    speed_torch = model.Speed(input_data_torch)
    speed_torch = torch.clip(speed_torch, 0, 1)
    valid_indice = (speed_torch < 0.9) & (speed_torch > 0.1)
    # pred_d = 1 - torch.sqrt(1-torch.sqrt(speed_torch))
    # pred_d = pred_d.detach().cpu().numpy()
    valid_indice = valid_indice.detach().cpu().numpy()
    pred_d = speed_torch.detach().cpu().numpy()
    return pred_d, valid_indice


def evaluate_sdf():
    pass


if __name__ == "__main__":
    #! load configs
    meshpath = "./data/auburn_scaled_5.obj"

    #gib 2
    src = np.array([0.25, 0.3, 0])
    tar = np.array([-0.15, -0.24, -0.05])
    #gib 7
    # src = np.array([0.15, 0.15, -0.04])
    # tar = np.array([-0.1, -0.3, 0.05])

    v, f = igl.read_triangle_mesh(meshpath)
    vertices=v
    faces=f
    t_obs = v[f].reshape(-1, 3)

    # x_bounds = [v[:,0].min()+0.5, v[:, 0].max()+0.5]
    # y_bounds = [v[:,1].min()+0.5, v[:, 1].max()+0.5]
    # z_bounds = [v[:,2].min()+0.5, v[:, 2].max()+0.5]


    # vertices = torch.tensor(vertices, dtype=torch.float32, device='cuda')
    # faces = torch.tensor(faces, dtype=torch.long, device='cuda')
    # triangles = vertices[faces].unsqueeze(dim=0)

    #! ################# Find Query Points
    # start_goal_file = os.path.join(root_path, "valid_start_goal.npz")
    # valid_points = np.load(start_goal_file)
    # # query_points = valid_points['start_points']
    N = 1000
    query_points = sample_points_inside_mesh(v, f, N)
    gt_tsdfs = gt_tsdf(t_obs, query_points)
    # #query_points = np.random.rand(100000, 3).astype(np.float32) - 0.5
    # #query_points *= 1.4



    #!###################
    #! GROUND TRUTH TSDF DISTANCE

    # query_points = torch.from_numpy(query_points).cuda()
    # bvh = bvh_distance_queries.BVH()
    # torch.cuda.synchronize()
    # torch.cuda.synchronize()
    # distances, closest_points, closest_faces, closest_bcs= bvh(triangles, query_points[None,])
    # torch.cuda.synchronize()
    # unsigned_distance = torch.sqrt(distances).squeeze()
    # unsigned_distance = unsigned_distance.detach().cpu().numpy()


    # query_points = query_points.detach().cpu().numpy()

    #! ############################ OUR METHOD ######################
    if True:
        #!
        #? Load the network
        experiment_folder = "/home/exx/Documents/FBPINNs/Experiments/08_15_09_59/"
        modelpath = "/home/exx/Documents/FBPINNs/Experiments/08_15_09_59/Model_Epoch_01700_ValLoss_1.055878e-02.pt"
        config_path = "configs/auburn.yaml"
        pred_d, valid_indice = our_tsdf(experiment_folder, modelpath, config_path, query_points)
        pred_d = pred_d[valid_indice]
        valid_gt_d = gt_tsdfs[valid_indice]

        err = np.abs(valid_gt_d - pred_d)
        print(np.mean(err))
        our_err = err

    #! ########################### iSDF #############################
    if False:
        from isdf.modules import trainer

        gib_id = 1
        chkpt_load_file = f"/home/n/iSDF/results/iSDF/gib_0{gib_id}/checkpoints/final.pth"
        
        config_file = "/home/n/iSDF/isdf/train/configs/ntfields.json"
        device = "cuda"
        # init trainer-------------------------------------------------------------
        isdf_trainer = trainer.Trainer(
            device,
            config_file,
            chkpt_load_file=chkpt_load_file,
            incremental=True,
            grid_dim = 200
        )
        sdf_map = isdf_trainer.sdf_map
        # s = 1/0.05937489
        # rand_samples *= s/2 (-0.5, 0.5)
        
        isdf_points = torch.from_numpy(query_points).cuda()
        pts = isdf_points
        pts *= 10
        # pts *= 1.2
        #with torch.set_grad_enabled(False):
        with torch.no_grad():
            pred_d = sdf_map(pts)
        pred_d = pred_d.detach().cpu().numpy()

        valid = pred_d>0
        pred_d = pred_d[valid]
        valid_gt_d = gt_tsdf[valid]
        # query_points = query_points[valid]
        # print(pred_d.min())
        err = np.abs(valid_gt_d - pred_d)
        print(np.mean(err))
        isdf_err = err

    #! ########################## Kinect Fusion #####################
    if False:
        grid = np.load("/home/n/KinectFusion/reconstruct/gibson1/kf_grid.npy")
        # grid[:,:,:75] = -1
        # grid[:,:,125:] = -1 
        grid = torch.from_numpy(grid[None, None, ...]).cuda()
        kf_points = torch.from_numpy(query_points).cuda()
        tmp = torch.zeros_like(kf_points)
        tmp[:,0]=kf_points[:,2]
        tmp[:,1]=kf_points[:,1]
        tmp[:,2]=kf_points[:,0]
        
        x = tmp
        p = x
    
        p = p.unsqueeze(0)
        p = p.unsqueeze(1).unsqueeze(1)
        #p *= 0.5
        #feature_0 = self.grid_sample_3d(f_0, p)
        pred_d = grid_sample_3d(grid, p).squeeze()
        pred_d = pred_d.detach().cpu().numpy()

        pred_d = np.clip(pred_d, minimum, maximum)/maximum
        valid = (pred_d>0) #& (pred_d<1)
        pred_d = pred_d[valid]
        valid_gt_d = gt_tsdf[valid]
        print(pred_d.min())
        err = np.abs(valid_gt_d - pred_d)
        # print(err)
        print(np.mean(err))
        kf_err = err

    # outputfile = "sdf_eval_gib1.npz"
    # np.savez(outputfile, our_err=our_err, isdf_err=isdf_err, kf_err=kf_err)
    # print(f"{outputfile} saved")
    #! ########################## vis
    if False: #! plot in trimesh 
        mesh = trimesh.load(meshpath)
        mesh.visual.face_colors = [200, 192, 207, 255]
        scene = trimesh.Scene(mesh)

        # Define line segments for X (red), Y (green), and Z (blue) axes
        axis_length = 1.0
        x_axis = trimesh.load_path(np.array([[0, 0, 0], [axis_length, 0, 0]]))
        y_axis = trimesh.load_path(np.array([[0, 0, 0], [0, axis_length, 0]]))
        z_axis = trimesh.load_path(np.array([[0, 0, 0], [0, 0, axis_length]]))
        x_axis.colors = [[255, 0, 0, 255]]
        y_axis.colors = [[0, 255, 0, 255]]
        z_axis.colors = [[0, 0, 255, 255]]
        scene.add_geometry([ x_axis, y_axis, z_axis])

        start_points = query_points
        # start_speeds = gt_tsdf 
        start_speeds = pred_d 
        colormap = plt.get_cmap('viridis')
        start_colors = colormap(start_speeds)

        point_cloud = trimesh.PointCloud(start_points, start_colors)
        scene.add_geometry([point_cloud])
        scene.show()


    
