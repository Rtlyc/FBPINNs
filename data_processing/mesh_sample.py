import numpy as np 
import torch
import igl
import matplotlib.pyplot as plt
import yaml

def viz(points, speeds, lidar_points=None, meshpath=None, camera_positions=None, plane=[-5, 5, -5, 5], height=0.0, scale=1.0, viz_start=True):
    #! visualize the mesh
    import trimesh
    if True:
        scene = trimesh.Scene()

        if meshpath:
            mesh = trimesh.load_mesh(meshpath)
            matrix = np.eye(4)
            matrix[:3, :3] *= scale

            mesh.apply_transform(matrix)
            mesh.visual.face_colors = [200, 192, 207, 255]
            scene.add_geometry([mesh])

        # Define line segments for X (red), Y (green), and Z (blue) axes
        axis_length = 1.0
        x_axis = trimesh.load_path(np.array([[0, 0, 0], [axis_length, 0, 0]]))
        y_axis = trimesh.load_path(np.array([[0, 0, 0], [0, axis_length, 0]]))
        z_axis = trimesh.load_path(np.array([[0, 0, 0], [0, 0, axis_length]]))
        x_axis.colors = [[255, 0, 0, 255]]
        y_axis.colors = [[0, 255, 0, 255]]
        z_axis.colors = [[0, 0, 255, 255]]
        scene.add_geometry([ x_axis, y_axis, z_axis])

        # Define camera positions
        if camera_positions is not None:
            cm_pc = trimesh.PointCloud(np.array(camera_positions), colors=[[255, 0, 0, 255]])
            scene.add_geometry([cm_pc])

        # Define a plane
        xmin, xmax, ymin, ymax = plane
        plane_vertices =[
            np.array([xmin, ymin, height]),
            np.array([xmax, ymin, height]),
            np.array([xmax, ymax, height]),
            np.array([xmin, ymax, height])
        ]
        plane_faces = [
            [0, 1, 2],
            [0, 2, 3],
            [0, 3, 2],
            [0, 2, 1]
        ]
        plane_mesh = trimesh.Trimesh(plane_vertices, plane_faces, process=False)
        plane_mesh.visual.face_colors = [100, 100, 255, 100]
        scene.add_geometry([plane_mesh])

        # add points cloud
        if True:
            if points is not None and (speeds is not None):
                start_points = points[:,:3]
                start_speeds = speeds[:, 0]
                # start_colors = (np.outer(1.0 - start_speeds, [255, 0, 0, 50]) + 
                #     np.outer(start_speeds, [255, 255, 255, 50])).astype(np.uint8)
                colormap = plt.get_cmap('viridis')
                start_colors = colormap(start_speeds)

                end_points = points[:,3:6]
                end_speeds = speeds[:, 1]
                colormap = plt.get_cmap('viridis')
                end_colors = colormap(end_speeds)
                # end_colors = (np.outer(1.0 - end_speeds, [255, 0, 0, 80]) + 
                #     np.outer(end_speeds, [255, 255, 255, 80])).astype(np.uint8)

                point_cloud = trimesh.PointCloud(start_points, start_colors)
                if not viz_start:
                    point_cloud = trimesh.PointCloud(end_points, end_colors)
                scene.add_geometry([point_cloud])
        

        if lidar_points is not None:
            points = lidar_points
            point_cloud = trimesh.PointCloud(points, colors=[255, 0, 0, 255])

            scene.add_geometry([ x_axis, y_axis, z_axis, point_cloud])


        scene.show()

def unsigned_distance_without_bvh(triangles, query_points):
    # Assuming your tensors are called triangles and query_points
    triangles_np = triangles
    query_points_np = query_points

    # Flatten and get unique vertices
    vertices, inverse_indices = np.unique(triangles_np, axis=0, return_inverse=True)

    # Convert back the inverse indices to get faces
    faces = inverse_indices.reshape(-1, 3)


    # Compute the squared distance (Note: to get the actual distance, take the sqrt of the results)
    squared_d, closest_faces, closest_points = igl.point_mesh_squared_distance(query_points_np, vertices, faces)

    # distances would be the sqrt of squared_d
    unsigned_distance = np.sqrt(squared_d)

    return unsigned_distance

def uniform_gt_pts_speeds(center, offsetxyz, limitxyz, wallsize=0, meshpath=None, minimum=0.07, maximum=0.3, num=10000, scale=1.0, region=None, must_inside=True):
    pc0 = torch.rand((num*10, 3))
    random_offsets = 2*torch.rand(num*10, 3)-1
    pc0[:,0] = random_offsets[:,0]*offsetxyz[0]+center[0]
    random_offsets = 2*torch.rand(num*10, 3)-1
    pc0[:,1] = random_offsets[:,1]*offsetxyz[1]+center[1]
    random_offsets = 2*torch.rand(num*10, 3)-1
    pc0[:,2] = random_offsets[:,2]*offsetxyz[2]+center[2]

    dP = torch.rand((num*10, 3))-0.5
    rL = torch.rand((num*10, 3))
    random_offsets = 2*torch.rand(num*10, 3)-1
    rL[:,0] = random_offsets[:,0]*offsetxyz[0]
    random_offsets = 2*torch.rand(num*10, 3)-1
    rL[:,1] = random_offsets[:,1]*offsetxyz[1]
    random_offsets = 2*torch.rand(num*10, 3)-1
    rL[:,2] = random_offsets[:,2]*offsetxyz[2]
    # rL[:,0] = rL[:,0]*0.3
    # rL[:,1] = rL[:,1]*0.3
    # rL[:,2] = rL[:,2]*0.3
    pc1 = pc0 + torch.nn.functional.normalize(dP, dim=1) * rL

    PointsInside = torch.all((pc1 <= limitxyz[0]), dim=1) & torch.all((pc1 >= -limitxyz[0]), dim=1)
    PointsInsidex = (pc1[:,0] <= center[0]+limitxyz[0]) & (pc1[:,0] >= center[0]-limitxyz[0])
    PointsInsidey = (pc1[:,1] <= center[1]+limitxyz[1]) & (pc1[:,1] >= center[1]-limitxyz[1])
    PointsInsidez = (pc1[:,2] <= center[2]+limitxyz[2]) & (pc1[:,2] >= center[2]-limitxyz[2])
    PointsInside = PointsInsidex & PointsInsidey & PointsInsidez
    #! only for cube_passage
    if False:
        # PointsInside = PointsInside & (torch.any(abs(pc0) > 1.0, dim=1) & torch.any(abs(pc1) > 1.0, dim=1))
        invalid_indices_0 = (pc0[:,0]>-2) & (pc0[:,0]<2.0) & (pc0[:,1]>-2.0) & (pc0[:,1]<2.0)
        invalid_indices_1 = (pc1[:,0]>-2) & (pc1[:,0]<2.0) & (pc1[:,1]>-2.0) & (pc1[:,1]<2.0)
        invalid_indices = invalid_indices_0 | invalid_indices_1
        PointsInside = PointsInside & ~invalid_indices

    if region is not None:
        PointsInside = PointsInside & (pc0[:,0] > region[0]) & (pc0[:,0] < region[1]) & (pc0[:,1] > region[2]) & (pc0[:,1] < region[3]) & (pc1[:,0] > region[0]) & (pc1[:,0] < region[1]) & (pc1[:,1] > region[2]) & (pc1[:,1] < region[3])

    pc0 = pc0[PointsInside]
    pc1 = pc1[PointsInside]


    v, f = igl.read_triangle_mesh(meshpath)
    v *= scale
    winding_numbers_0 = igl.winding_number(v, f, pc0.cpu().numpy())
    winding_numbers_1 = igl.winding_number(v, f, pc1.cpu().numpy())
    PointsInside = (abs(winding_numbers_0) > 0.1) & (abs(winding_numbers_1) > 0.1)
    pc0 = pc0[PointsInside]
    pc1 = pc1[PointsInside]
    t_obs = v[f].reshape(-1, 3)

    device = pc0.device
    x0 = pc0.cpu().numpy()
    y0_bounds = unsigned_distance_without_bvh(t_obs, x0)
    y0_bounds -= wallsize
    y0_bounds = torch.from_numpy(y0_bounds).float().to(device).unsqueeze(1)
    # bounds to speeds
    # speeds = torch.clip(bounds, minimum, maximum)/maximum
    y0 = torch.clip(y0_bounds, minimum, maximum)/maximum

    valid_indices = torch.where((y0 < 1) & (y0 > 0))[0]
    if num <= len(valid_indices):
        # Select without replacement if num is less than or equal to the size of valid_indices
        start_indices = valid_indices[torch.randperm(len(valid_indices))[:num]]
    else:
        # Select with replacement if num is greater than the size of valid_indices
        rand_indices = torch.randint(0, len(valid_indices), (num,))
        start_indices = valid_indices[rand_indices]

    x1 = pc1[start_indices].cpu().numpy()
    y1_bounds = unsigned_distance_without_bvh(t_obs, x1)
    y1_bounds -= wallsize
    y1_bounds = torch.from_numpy(y1_bounds).float().to(device).unsqueeze(1)
    # y1 = torch.clip((y1 - minimum) / (maximum - minimum), 0, 1)
    y1 = torch.clip(y1_bounds, minimum, maximum)/maximum

    x0 = pc0[start_indices]
    x1 = pc1[start_indices]
    x = torch.cat((x0, x1), dim=1)
    y = torch.cat((y0[start_indices], y1), dim=1)
    z = torch.cat((y0_bounds[start_indices], y1_bounds), dim=1)

    # import matplotlib.pyplot as plt
    # plt.scatter(all_bounds, all_speeds)
    # plt.xlabel("ground truth distance")
    # plt.ylabel("speed")
    # plt.show()
    # temp = speeds[start_indices]
    # plt.hist(temp.cpu().numpy(), bins=100)
    # plt.title("speed distribution")
    # plt.show()
    return x, y, z


if __name__ == '__main__':
    import sys
    import yaml
    # # meshpath = "mesh_scaled_11_29_15.obj"
    # meshpath = "data/passage2.obj"
    # meshpath = "data/cube_passage.obj"
    # meshpath = "data/cabin_mesh.obj"
    # sample_number = 200000
    # dim = 3
    # # sample_speed(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
    # # points, speeds = my_sample_speed(meshpath, sample_number, dim)
    # center = np.array([0, 0, 0])
    # offset = np.array([2.0, 10.0, 2.0])
    # offset = np.array([3.0, 3.0, 3.0]) 
    # offset = np.array([1.0, 1.0, 1.0]) 
    # minimum = 0.008
    # maximum = 0.08

    #! load configs
    config_path = "configs/cabin.yaml"
    config_path = "configs/maze.yaml"
    config_path = "configs/ruiqi.yaml"
    config_path = "configs/mesh.yaml"
    config_path = "configs/cube_passage.yaml"
    config_path = "configs/almena.yaml"
    config_path = "configs/narrow_cube.yaml"
    config_path = "configs/auburn.yaml"


    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    center = np.array(config["data"]["center"])
    offset = np.array(config["data"]["offset"])
    limit = np.array(config["data"]["limit"])
    wallsize = config["data"]["wallsize"]
    dim = config["data"]["dim"]
    minimum = config["data"]["minimum"]
    maximum = config["data"]["maximum"]
    sample_number = config["data"]["sample_num"]
    scale = config["data"]["scaling"]
    meshpath = config["paths"]["meshpath"]
    # pointspath = config["paths"]["pointspath"]
    # speedspath = config["paths"]["speedspath"]
    name = config["paths"]["name"]

    #? 2D window
    region_predefined = config["region"]["use_predefined"]
    columns, rows = 3, 3
    if True:
        regions = config["region"]['regions']
    else:
        regions = []
        xmin, xmax, ymin, ymax = config["region"]["boundaries"]
        columns = config["region"]["columns"]
        rows = config["region"]["rows"]
        overlap_ratio = config["region"]["overlap_ratio"]

        width_total = xmax - xmin
        height_total = ymax - ymin
        width_core = width_total / (columns)
        width_overlap = width_core * overlap_ratio
        column_regions = [(xmin - width_overlap + i*width_core, xmin + width_overlap + i*width_core) for i in range(columns)]

        height_core = height_total / (rows)
        height_overlap = height_core * overlap_ratio
        row_regions = [(ymin - height_overlap + i*height_core, ymin + height_overlap + i*height_core) for i in range(rows)]

        for i in range(rows):
            for j in range(columns):
                region = (column_regions[j][0], column_regions[j][1], row_regions[i][0], row_regions[i][1])
                regions.append(region)


    if False: #? ground truth points viz
        explored_data = np.load("data/explored_data.npy").reshape(-1, 8)
        viz(explored_data[:, :6], explored_data[:, 6:], meshpath=None, scale=scale)
        points = np.load("data/ruiqi_points.npy")
        speeds = np.load("data/ruiqi_speeds.npy")
        viz(points, speeds, meshpath=None, scale=scale)

    # if True:
    #     explored_data = np.load("data/explored_data.npy")
    #     #! hardcode room size
    #     explored_data[:, :, :6] *= 5.0
    #     print("hardcoded room size scale: 5.0")
    # else:
    #     points = np.load("data/passage_points.npy")
    #     speeds = np.load("data/passage_speeds.npy")
    #     explored_data = np.concatenate((points, speeds), axis=1)
    #     explored_data = explored_data.reshape(-1, 2000, 8)
    #     explored_data[:, :, :6] *= 1/5.0
    # explored_data = explored_data.reshape(-1, 8)
    # points = explored_data[:, :6]
    # speeds = explored_data[:, 6:]
    for ind, region in enumerate(regions):
        points, speeds, bounds = uniform_gt_pts_speeds(center, offset, limit, wallsize, meshpath, minimum, maximum, sample_number, scale=scale, region=None)
        pointspath = f"data/{name}_points_{ind}.npy"
        speedspath = f"data/{name}_speeds_{ind}.npy"
        boundspath = f"data/{name}_bounds_{ind}.npy"
        print("points shape: ", points.shape)
        np.save(pointspath, points)
        np.save(speedspath, speeds)
        np.save(boundspath, bounds)
        if True:
            plane = region
            rand_indices = torch.randint(0, len(points), (10000,))
            points = points[rand_indices]
            speeds = speeds[rand_indices]
            viz(points, speeds, meshpath=None, scale=scale, plane=plane, height=center[2])
            viz(points, speeds, meshpath=meshpath, scale=scale, plane=plane, height=center[2], viz_start=False)
            print()
    # region = [0.5, 3.5, 0.5, 3.5]
    # ind = 6
    # points, speeds = uniform_gt_pts_speeds(center, offset, meshpath, minimum, maximum, sample_number, scale=scale, region=region)


    # if True:
    #     # np.save("data/passage_points.npy", points)
    #     # np.save("data/passage_speeds.npy", speeds)
    #     # np.save("data/cube_passage_points.npy", points)
    #     # np.save("data/cube_passage_speeds.npy", speeds)
    #     # np.save("data/cabin_points_05.npy", points)
    #     # np.save("data/cabin_speeds_05.npy", speeds)
    #     pointspath = f"data/cube_passage_points_{ind}.npy"
    #     speedspath = f"data/cube_passage_speeds_{ind}.npy"
    #     print("points shape: ", points.shape)
    #     np.save(pointspath, points)
    #     np.save(speedspath, speeds)
