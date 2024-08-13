import numpy
import matplotlib.pyplot as plt
import math
import torch
import matplotlib.patches as patches
import os


def generate_possible_blocks(valid_indices, block_mapping, col, row):
    R, C = row, col
    blocks = []
    
    for j1 in range(C):
        for i1 in range(R):
            if [j1, i1] in valid_indices.tolist():
                for j2 in range(j1, C):
                    for i2 in range(i1, R):
                        if all((j, i) in valid_indices.tolist() or block_mapping[j, i] != 1000 for j in range(j1, j2+1) for i in range(i1, i2+1)):
                            blocks.append((j1, i1, j2, i2))
    return blocks

def filter_blocks(blocks, block_mapping):
    valid_blocks = []
    
    for (j1, i1, j2, i2) in blocks:
        total_cells = (j2 - j1 + 1) * (i2 - i1 + 1)
        occupied_cells = sum(1 for j in range(j1, j2+1) for i in range(i1, i2+1) if block_mapping[j, i] == 1000)
        
        if occupied_cells / total_cells <= 0.25:
            valid_blocks.append((j1, i1, j2, i2))
    
    return valid_blocks
    


def select_minimum_blocks(valid_blocks, valid_indices):
    uncovered = set(tuple(map(tuple, valid_indices.tolist())))
    selected_blocks = []
    
    while uncovered:
        best_block = None
        best_cover = set()
        min_block_size = float('inf')
        
        for block in valid_blocks:
            j1, i1, j2, i2 = block
            block_cover = {(j, i) for j in range(j1, j2+1) for i in range(i1, i2+1) if (j, i) in uncovered}
            block_size = (j2 - j1 + 1) * (i2 - i1 + 1)
            
            # Choose block with the best cover; if two covers are equal, choose the smaller block
            if len(block_cover) > len(best_cover) or (len(block_cover) == len(best_cover) and block_size < min_block_size):
                best_block = block
                best_cover = block_cover
                min_block_size = block_size
        
        if best_block:
            selected_blocks.append(best_block)
            uncovered -= best_cover
    
    return selected_blocks



def blocks_to_regions(minimum_blocks, block_idx_to_subnet_idx, regions):
    subnets = []
    new_block_idx_to_subnet_idx = torch.ones_like(block_idx_to_subnet_idx) * 1000
    for j1, i1, j2, i2 in minimum_blocks:
        subnet0 = block_idx_to_subnet_idx[j1, i1]
        subnet1 = block_idx_to_subnet_idx[j2, i2]
        x_min, x_max = min(regions[subnet0][0], regions[subnet1][0]), max(regions[subnet0][1], regions[subnet1][1])
        y_min, y_max = min(regions[subnet0][2], regions[subnet1][2]), max(regions[subnet0][3], regions[subnet1][3])
        new_subnet = (x_min, x_max, y_min, y_max)
        new_block_idx_to_subnet_idx[j1:j2+1, i1:i2+1] = len(subnets)
        subnets.append(new_subnet)
        #*: think about new block_idx_to_subnet_idx
    return subnets, new_block_idx_to_subnet_idx




class OccupancyGridMap:
    def __init__(self, data_array, cell_size, occupancy_threshold=0.8, offset=0.5, block_cols=4, block_rows=4):
        """
        Creates a grid map
        :param data_array: a 2D array with a value of occupancy per cell (values from 0 - 1)
        :param cell_size: cell size in meters
        :param occupancy_threshold: A threshold to determine whether a cell is occupied or free.
        A cell is considered occupied if its value >= occupancy_threshold, free otherwise.
        """

        self.data = data_array
        self.dim_cells = data_array.shape
        self.dim_meters = (self.dim_cells[0] * cell_size, self.dim_cells[1] * cell_size)
        self.cell_size = cell_size
        self.occupancy_threshold = occupancy_threshold
        # 2D array to mark visited nodes (in the beginning, no node has been visited)
        self.visited = numpy.zeros(self.dim_cells, dtype=numpy.float32)
        self.offset = offset

        # Stack to store the blocks of the map that have not been visited well
        self.block_cols = block_cols
        self.block_rows = block_rows
        # self.block_cells = (self.block_cols, self.block_rows)
        # self.block_cells = (self.dim_cells[0]//4, self.dim_cells[1]//4)
        self.per_col_pixels = self.dim_cells[0]//self.block_cols
        self.per_row_pixels = self.dim_cells[1]//self.block_rows
        self.per_col_size = self.per_col_pixels * self.cell_size
        self.per_row_size = self.per_row_pixels * self.cell_size

        self.overlap_ratio = 1/4
        # self.block_meters = self.cell_size * 4
        # self.block_stack = []

    def save(self, filepath):
        numpy.savez(filepath, data=self.data, visited=self.visited) 

    def load(self, filepath):
        f = numpy.load(filepath)
        self.data = f['data']
        self.visited = f['visited']
        

    def mark_visited_idx(self, point_idx, status=1):
        """
        Mark a point as visited with a status.
        :param point_idx: a point (x, y) in data array
        :param status: 1 for bad visited, 2 for good visited
        """
        x_index, y_index = point_idx
        if not self.is_inside_idx((x_index, y_index)):
            raise Exception('Point is outside map boundary')

        self.visited[y_index][x_index] = status

    def mark_visited(self, point, status):
        """
        Mark a point as visited.
        :param point: a 2D point (x, y) in meters
        """
        x, y = point
        x_index, y_index = self.get_index_from_coordinates(x, y)

        return self.mark_visited_idx((x_index, y_index), status)

        
    def is_visited_idx(self, point_idx):
        """
        Check whether the given point is visited.
        :param point_idx: a point (x, y) in data array
        :return: Status of the visit (0, 1, or 2)
        """
        x_index, y_index = point_idx
        if not self.is_inside_idx((x_index, y_index)):
            raise Exception('Point is outside map boundary')

        return self.visited[y_index][x_index]

    def is_visited(self, point):
        """
        Check whether the given point is visited.
        :param point: a 2D point (x, y) in meters
        :return: True if the given point is visited, false otherwise
        """
        x, y = point
        x_index, y_index = self.get_index_from_coordinates(x, y)

        return self.is_visited_idx((x_index, y_index))

    def get_data_idx(self, point_idx):
        """
        Get the occupancy value of the given point.
        :param point_idx: a point (x, y) in data array
        :return: the occupancy value of the given point
        """
        x_index, y_index = point_idx
        if x_index < 0 or y_index < 0 or x_index >= self.dim_cells[0] or y_index >= self.dim_cells[1]:
            raise Exception('Point is outside map boundary')

        return self.data[y_index][x_index]

    def get_data(self, point):
        """
        Get the occupancy value of the given point.
        :param point: a 2D point (x, y) in meters
        :return: the occupancy value of the given point
        """
        x, y = point
        x_index, y_index = self.get_index_from_coordinates(x, y)

        return self.get_data_idx((x_index, y_index))

    def set_data_idx(self, point_idx, new_value):
        """
        Set the occupancy value of the given point.
        :param point_idx: a point (x, y) in data array
        :param new_value: the new occupancy values
        """
        x_index, y_index = point_idx
        if x_index < 0 or y_index < 0 or x_index >= self.dim_cells[0] or y_index >= self.dim_cells[1]:
            raise Exception('Point is outside map boundary')

        self.data[y_index][x_index] = new_value

    def set_data(self, point, new_value):
        """
        Set the occupancy value of the given point.
        :param point: a 2D point (x, y) in meters
        :param new_value: the new occupancy value
        """
        x, y = point
        x_index, y_index = self.get_index_from_coordinates(x, y)

        self.set_data_idx((x_index, y_index), new_value)

    def is_inside_idx(self, point_idx):
        """
        Check whether the given point is inside the map.
        :param point_idx: a point (x, y) in data array
        :return: True if the given point is inside the map, false otherwise
        """
        x_index, y_index = point_idx
        if x_index < 0 or y_index < 0 or x_index >= self.dim_cells[0] or y_index >= self.dim_cells[1]:
            return False
        else:
            return True

    def is_inside(self, point):
        """
        Check whether the given point is inside the map.
        :param point: a 2D point (x, y) in meters
        :return: True if the given point is inside the map, false otherwise
        """
        x, y = point
        x_index, y_index = self.get_index_from_coordinates(x, y)

        return self.is_inside_idx((x_index, y_index))

    def is_occupied_idx(self, point_idx):
        """
        Check whether the given point is occupied according the the occupancy threshold.
        :param point_idx: a point (x, y) in data array
        :return: True if the given point is occupied, false otherwise
        """
        x_index, y_index = point_idx
        if self.get_data_idx((x_index, y_index)) >= self.occupancy_threshold:
            return True
        else:
            return False

    def is_occupied(self, point):
        """
        Check whether the given point is occupied according the the occupancy threshold.
        :param point: a 2D point (x, y) in meters
        :return: True if the given point is occupied, false otherwise
        """
        x, y = point
        x_index, y_index = self.get_index_from_coordinates(x, y)

        return self.is_occupied_idx((x_index, y_index))

    def get_index_from_coordinates(self, x, y):
        """
        Get the array indices of the given point.
        :param x: the point's x-coordinate in meters
        :param y: the point's y-coordinate in meters
        :return: the corresponding array indices as a (x, y) tuple
        """
        # print("get_index_from_coordinates pre: ", (x, y))
        x = x + self.offset
        y = y + self.offset
        # print("get_index_from_coordinates: ", (x, y))
        # print("cell_size: ", self.cell_size)
        x_index = math.floor(x/self.cell_size)
        y_index = math.floor(y/self.cell_size)

        return x_index, y_index

    def get_coordinates_from_index(self, x_index, y_index):
        """
        Get the coordinates of the given array point in meters.
        :param x_index: the point's x index
        :param y_index: the point's y index
        :return: the corresponding point in meters as a (x, y) tuple
        """
        x = x_index*self.cell_size
        y = y_index*self.cell_size
        x = x - self.offset
        y = y - self.offset
        return x, y

    def get_all_free_space_indices(self):
        mask = (self.visited == 2) & (self.data < self.occupancy_threshold)
        indices = numpy.where(mask)
        return indices
        
    def get_all_free_space_coordinates(self):
        y_indices, x_indices = self.get_all_free_space_indices()
        coordinates = []
        for i in range(len(x_indices)):
            x, y = x_indices[i], y_indices[i]
            coordinates.append(self.get_coordinates_from_index(x, y))
        return numpy.array(coordinates)

    def plot(self, curloc, end_points, alpha=1, origin='lower', path=None):
        """
        Plot the grid map with different colors based on visit status.
        """
        # Create a color map: 0 - grey, 1 - red, 2 - black/white based on occupancy
        colored_map = numpy.empty((*self.dim_cells, 3))
        for j in range(self.dim_cells[0]):
            for i in range(self.dim_cells[1]):
                # if self.visited[j, i] == 0:  # not visited
                #     colored_map[j, i] = [0.5, 0.5, 0.5]  # grey
                # elif self.visited[j, i] == 1:  # bad visited
                #     colored_map[j, i] = [1, 0, 0]  # red
                # else:  # good visited
                #     if self.data[j, i] >= self.occupancy_threshold:
                #         colored_map[j, i] = [0, 0, 0]  # black for occupied
                #     else:
                #         colored_map[j, i] = [1, 1, 1]  # white for not occupied
                if self.visited[j, i] == 0: # not visited
                    colored_map[j, i] = [0.5, 0.5, 0.5] # grey
                elif self.data[j, i] >= self.occupancy_threshold:
                    colored_map[j, i] = [0, 0, 0] # black for occupied
                else:
                    colored_map[j, i] = [1, 1, 1] # white for not occupied

        
        # for current location and target location
        curi,curj = self.get_index_from_coordinates(curloc[0],curloc[1])
        print("curij: ", curi, curj)    
        colored_map[curj,curi] = [138/255,43/255,226/255]
        # targeti,targetj = self.get_index_from_coordinates(targetloc[0],targetloc[1])
        # print("targetij: ", targeti, targetj)
        # colored_map[targetj,targeti] = [255/255,20/255,147/255]

        # if True: # plot the centers of the blocks
        #     centers = self.get_block_centers()
        #     for center in centers:
        #         centeri,centerj = self.get_index_from_coordinates(center[0],center[1])
        #         colored_map[centerj,centeri] = [0,1,1]


        # Create a new RGBA image to hold the overlay
        overlay = numpy.zeros((*self.dim_cells, 4))

        # Highlight blocks in block_stack by drawing rectangles
        # print("block_stack: ", self.block_stack)
        # print("block_stack_size: ", len(self.block_stack))
        # end_points = torch.tensor([[-3, -2.5]])
        valid_indices, map_indices = self.filter_points(end_points)
        end_points = end_points[valid_indices]
        mask = self.get_region_points_mask(end_points) #(12, 50000)
        # get to know sum of mask to (50000, 1)
        neighbor_counts = torch.sum(mask, dim=0).cpu().numpy()
        map_indices = map_indices.cpu().numpy()
        four_colors = numpy.array([[47, 243, 224, 130], [248, 210, 16, 130], [250, 38, 160, 130], [244, 23, 32, 130]])/255
        overlay[map_indices[:, 1], map_indices[:, 0], :] = four_colors[neighbor_counts[:,]-1] 
        print()



        # for block_idx in self.valid_blocks_indices:
        #     col, row = block_idx
        #     overlay[col*4:(col+1)*4, row*4:(row+1)*4, :] = [0, 1, 0, 0.5]  # Highlighted in green with alpha

        # Combine the original colored_map and the overlay
        final_image = colored_map.copy()
        final_image = final_image.reshape(final_image.shape[0], final_image.shape[1], -1)
        overlay = overlay.reshape(overlay.shape[0], overlay.shape[1], -1)
        final_image = (1 - overlay[:, :, 3:4]) * final_image + overlay[:, :, :3] * overlay[:, :, 3:4]

        fig, ax = plt.subplots()
        # Set the extent of the axes if needed
        extent = [0, self.dim_cells[1], 0, self.dim_cells[0]]  # [xmin, xmax, ymin, ymax]
        ax.imshow(final_image, origin=origin, interpolation='none', alpha=alpha, extent=extent)

        # ax.imshow(final_image, origin=origin, interpolation='none', alpha=alpha)

        plt.draw()
        if path is not None:
            figpath = os.path.join(path, 'occ_map.png')
            plt.savefig(figpath)
        plt.close()


        #! draw the all blocks
        # colored_map = numpy.empty((*self.dim_cells, 3))
        # possible_regions = self.get_batch_block_indices(end_points)
        # origin_regions = possible_regions[:, 0, :].cpu().numpy()
        # fig, ax = plt.subplots()
        # # Set the extent of the axes if needed
        # extent = [0, self.dim_cells[1], 0, self.dim_cells[0]]  # [xmin, xmax, ymin, ymax]
        # ax.imshow(final_image, origin=origin, interpolation='none', alpha=alpha, extent=extent)

        # ax.imshow(final_image, origin=origin, interpolation='none', alpha=alpha)

        # plt.draw()
        # if path is not None:
        #     plt.savefig(path)
        # plt.close()



        # Initialize a plot
        fig, ax = plt.subplots()

        # Generate a colormap with distinct colors for each region
        cmap = plt.get_cmap('tab20')  # Use a colormap with enough distinct colors
        colors = cmap(numpy.linspace(0, 1, len(self.regions)))

        # Draw each region as a rectangle with a different color
        for i, region in enumerate(self.regions):
            # Convert the actual (x_min, y_min) and (x_max, y_max) to indices
            x_min, x_max, y_min, y_max = region
            row_min, col_min = self.get_index_from_coordinates(x_min, y_min)
            row_max, col_max = self.get_index_from_coordinates(x_max, y_max)

            # Create a rectangle patch
            rect = patches.Rectangle((row_min, col_min), row_max - row_min, col_max - col_min,
                                    linewidth=1, edgecolor='black', facecolor=colors[i], alpha=0.5)

            # Add the rectangle to the plot
            ax.add_patch(rect)

        # Set the extent of the axes
        extent = [0, self.dim_cells[1], 0, self.dim_cells[0]]  # [xmin, xmax, ymin, ymax]
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])

        # Optionally set other plot properties
        ax.set_aspect('equal', adjustable='box')

        # Draw the plot
        plt.draw()

        # Save the figure if a path is provided
        if path is not None:
            figpath = os.path.join(path, 'occ_map_regions.png')
            plt.savefig(figpath)

        # Close the plot to avoid displaying it immediately in certain environments
        plt.close()


    def get_batch_block_indices(self, points):
        #?: need to check col, row matchings
        # get current block indices and neighbor indices with a 25% overlap
        indices = self.get_batch_indices(points)
        x_indices = indices[:, 0]
        y_indices = indices[:, 1]
        block_x_indices = x_indices // self.per_col_pixels
        block_y_indices = y_indices // self.per_row_pixels

        rest_x_indices = x_indices % self.per_col_pixels
        rest_y_indices = y_indices % self.per_row_pixels
        rest_x_indices_percentage = rest_x_indices / self.per_col_pixels
        rest_y_indices_percentage = rest_y_indices / self.per_row_pixels
        rest_x_indices_neighbor = rest_x_indices_percentage - 0.5
        rest_x_indices_neighbor = torch.where(rest_x_indices_neighbor < -0.25, -1, torch.where(rest_x_indices_neighbor > 0.25, 1, 0))
        rest_y_indices_neighbor = rest_y_indices_percentage - 0.5
        rest_y_indices_neighbor = torch.where(rest_y_indices_neighbor < -0.25, -1, torch.where(rest_y_indices_neighbor > 0.25, 1, 0))

        possible_regions = torch.ones((points.shape[0], 4, 2), device=points.device) # (50000, 4, 2)
        possible_regions *= -1
        possible_regions[:, 0] = torch.stack([block_x_indices, block_y_indices], dim=1)
        possible_regions[:, 1] = torch.stack([block_x_indices + rest_x_indices_neighbor, block_y_indices], dim=1)
        possible_regions[:, 2] = torch.stack([block_x_indices, block_y_indices + rest_y_indices_neighbor], dim=1)
        possible_regions[:, 3] = torch.stack([block_x_indices + rest_x_indices_neighbor, block_y_indices + rest_y_indices_neighbor], dim=1)

        #! will return a raw (50000, 4, 2) tensor, need check for out of boundary, also need to check for the valid block stack
        possible_regions[:, :, 0] = torch.clamp(possible_regions[:, :, 0], 0, self.block_cols-1)
        possible_regions[:, :, 1] = torch.clamp(possible_regions[:, :, 1], 0, self.block_rows-1)
        return possible_regions.long()

        # valid_indices = (possible_regions[:, :, 0] < self.block_cells[0]) & (possible_regions[:, :, 0] >= 0) & (possible_regions[:, :, 1] < self.block_cells[1]) & (possible_regions[:, :, 1] >= 0)
        # return torch.stack([block_x_indices, block_y_indices, rest_x_indices_neighbor, rest_y_indices_neighbor], dim=1)

    def get_region_points_mask(self, end_points):
        #!: use col, row indicing some array
        possible_regions = self.get_batch_block_indices(end_points)
        # valid_blocks_indices = torch.tensor(valid_blocks_indices, device=possible_regions.device)
        # M, N = valid_blocks_indices.shape[0], possible_regions.shape[0]
        # Expand dimensions for broadcasting
        # expanded_valid_blocks = valid_blocks_indices.unsqueeze(1).unsqueeze(2).expand(M, N, 4, 2)
        # expanded_possible_regions = possible_regions.unsqueeze(0).expand(M, N, 4, 2)

        # Compare and check if any of the 4 coordinates match the valid blocks
        # matches = torch.all(expanded_valid_blocks == expanded_possible_regions, dim=-1)

        # Create mask by checking if there is any match in the 4 coordinates
        # mask = torch.any(matches, dim=-1)

        #! check with self.block_idx_to_subnet_idx
        M = len(self.regions)
        N = possible_regions.shape[0]
        # expanded_possible_regions = possible_regions.unsqueeze(0).expand(M, N, 4, 2)
        # expanded_regions = self.block_idx_to_subnet_idx

        x_indices = possible_regions[..., 0]
        y_indices = possible_regions[..., 1]
        block_idx_to_subnet_idx = self.block_idx_to_subnet_idx.to(x_indices.device)
        subnet_indices = block_idx_to_subnet_idx[x_indices, y_indices] #TODO: check the order of x, y
        mask = torch.zeros(M, N, dtype=torch.bool, device=subnet_indices.device)

        for i in range(M):
            mask[i] = torch.any(subnet_indices == i, dim=1)


        return mask

    def filter_points(self, points):
        indices = self.get_batch_indices(points) # (i, j)
        valid_indices = (indices[:, 0] >= 0) & (indices[:, 0] < self.dim_cells[0]) & (indices[:, 1] >= 0) & (indices[:, 1] < self.dim_cells[1])
        return valid_indices, indices[valid_indices]
    
    def get_batch_indices(self, points):
        offset_points = points + self.offset 
        indices = (offset_points / self.cell_size).floor().long()
        return indices
    


    def update(self, position, end_points, end_bounds):
        """
        Update the grid map with the given frame points and bounds.
        :param frame_points: a list of points in the frame
        :param frame_bounds: the bounds of the frame

        Update strategy:
        1. Mark all points in the frame as visited, if they are far away from the robot, mark them as bad visited, otherwise good visited
        2. calculate the occupancy value of the points in the frame based on the bounds.If bounds smaller, we assume occupancy values are larger. More Specifically: bounds -> occupancy, [-1, 1] -> [0, 1], bounds valid range(0.01, 0.1), 0.01 -> 1, 0.1 -> 0
        """

        """* :
            1. Given frame_points and frame_bounds, update all points as visited 
            2. Update the occupancy value based on the bounds
        """
        #TODO: May need to consider z value
        # end_points = frame_points[:, 3:5]
        # end_bounds = frame_bounds[:, 1:]

        valid_indices, map_indices = self.filter_points(end_points)
        # invalid_indices = (indices[:, 0] < 0) | (indices[:, 0] >= self.dim_cells[0]) | (indices[:, 1] < 0) | (indices[:, 1] >= self.dim_cells[1])
        # indices = indices[~invalid_indices]
        end_points = end_points[valid_indices]
        end_bounds = end_bounds[valid_indices]

        minimum = 0.04
        maximum = 0.4
        end_speeds = torch.clamp(end_bounds, minimum, maximum)/maximum
        end_occupancies = 1 - end_speeds

        # indices = torch.clamp(indices, 0, self.dim_cells[0]-1)
        map_indices = map_indices.cpu().numpy()
        self.visited[map_indices[:, 1], map_indices[:, 0]] = 2
        self.data[map_indices[:, 1], map_indices[:, 0]] = end_occupancies[:, 0].cpu().numpy()

        #!: create valid block stack
        blocks = self.data.reshape(self.block_cols, self.per_col_pixels, self.block_rows, self.per_row_pixels)
        pixels_free_rate = ((blocks < self.occupancy_threshold).sum(axis=(1, 3)))/(self.per_col_pixels*self.per_row_pixels)
        self.valid_blocks_indices = numpy.argwhere(pixels_free_rate > 0.2) # (j, i)
        
        

        #?1. naive algorithm, show all avaliable blocks directly
        self.regions = []
        xmin, xmax, ymin, ymax = -self.offset, self.dim_meters[0]-self.offset, -self.offset, self.dim_meters[1]-self.offset
        columns = self.block_cols
        rows = self.block_rows
        overlap_ratio = self.overlap_ratio

        width_total = xmax - xmin
        height_total = ymax - ymin
        
        width_core = width_total / (columns)
        width_overlap = width_core * overlap_ratio
        column_regions = [(xmin - width_overlap + i*width_core, xmin + width_core+width_overlap + i*width_core) for i in range(columns)]

        height_core = height_total / (rows)
        height_overlap = height_core * overlap_ratio
        row_regions = [(ymin - height_overlap + i*height_core, ymin + height_overlap+height_core + i*height_core) for i in range(rows)]

        self.block_idx_to_subnet_idx = 1000*torch.ones((columns, rows), dtype=torch.int32)
        for j, i in self.valid_blocks_indices:
            region = (column_regions[j][0], column_regions[j][1], row_regions[i][0], row_regions[i][1])
            self.regions.append(region)
            self.block_idx_to_subnet_idx[j, i] = len(self.regions) - 1

        #?2. some merge algorithm, merge to larger fewer blocks
        if True:
            # Generate all possible blocks considering non-square grid
            possible_blocks = generate_possible_blocks(self.valid_blocks_indices, self.block_idx_to_subnet_idx, self.block_cols, self.block_rows)

            # Filter out invalid blocks that don't meet the 25% occupancy rule
            valid_blocks = filter_blocks(possible_blocks, self.block_idx_to_subnet_idx)

            # Select the minimum number of blocks to cover all free grids
            minimum_blocks = select_minimum_blocks(valid_blocks, self.valid_blocks_indices)
            self.regions, self.block_idx_to_subnet_idx = blocks_to_regions(minimum_blocks, self.block_idx_to_subnet_idx, self.regions)
            print()




        # #?: get region points mask
        # region_points_mask = self.get_region_points_mask(end_points, self.valid_blocks_indices)
        # print()



        # for i in range(end_points.shape[0]):
        #     cur_point = end_points[i]
        #     cur_bound = end_bounds[i][0]

        #     # Mark the point as visited
        #     dist = numpy.linalg.norm(cur_point - position)
        #     if dist > 0.2: # bad visited distance
        #         if self.is_visited(cur_point[:2]) == 0:
        #             self.mark_visited(cur_point[:2], 1) # mark as bad visited
        #     else:
        #         self.mark_visited(cur_point[:2], 2) # mark as good visited

        #     # Update the occupancy value of the point
        #     # speeds = torch.clip((bounds - minimum) / (maximum - minimum), 0.1, 1)
        #     minimum = 0.01 # value we assume to be obstacle
        #     maximum = 0.1 # value we assume to be free
        #     occ_val = 1 - numpy.clip((cur_bound - minimum) / (maximum - minimum), 0, 1)
        #     self.set_data(cur_point[:2], occ_val)

        # #?: Update the block stack, some hardcoding here, need to be tuned
        # blocks = self.visited.reshape(self.block_cells[0], 4, self.block_cells[1], 4)
        # pixels_not_well_visited_rate = ((blocks == 1).sum(axis=(1, 3)))/16
        # indices_not_well_visited = numpy.argwhere(pixels_not_well_visited_rate > 0.25)
        # # Swap the columns to get indices in [j, i] format
        # # indices_not_well_visited = indices_not_well_visited[:, [1, 0]]
        # self.block_stack = indices_not_well_visited.tolist()

    # def get_target_block(self):
    #     """
    #     Get the target block to explore.
    #     :return: the target block
    #     """
    #     #?: some hardcoding here, need to be tuned
    #     if len(self.block_stack) == 0:
    #         return None
    #     else:
    #         cur_block = self.block_stack.pop(0)
    #         cur_data = self.get_block_data(cur_block)
    #         while numpy.sum(cur_data == 1)/16 < 0.25:
    #             if len(self.block_stack) == 0:
    #                 return None
    #             else:
    #                 cur_block = self.block_stack.pop(0)
    #                 cur_data = self.get_block_data(cur_block)
    #         return cur_block

    def get_regions(self):
        return self.regions, self.block_idx_to_subnet_idx
    

    def get_block_data(self, block):
        """
        Get the data of the given block.
        :param block: the block to get data from
        :return: the data of the given block
        """
        # ?: some hardcoding here, need to be tuned
        x_index, y_index = block
        return self.visited[x_index*4:(x_index+1)*4, y_index*4:(y_index+1)*4]
    
    def get_block_center(self, block):
        """
        Get the center of the given block.
        :param block: the block to get center from
        :return: the center of the given block
        """
    
        y_index, x_index = block
        #temp_x = x_index*4 + 2
        #temp_y = y_index*4 + 2
        #x,y = self.get_coordinates_from_index(temp_x, temp_y)
        for i in range (-2,6):
            for j in range (-2,6):
                temp_x = x_index*4 + i
                temp_y = y_index*4 + j
                #print(self.visited[temp_y, temp_x])
                if temp_x>=0 and temp_y>=0 and temp_x<=99 and temp_y<=99:
                    if self.visited[temp_y, temp_x] == 2 and self.data[temp_y, temp_x] < self.occupancy_threshold:
                        x, y = self.get_coordinates_from_index(temp_x, temp_y)
                        return x, y
        # x = (x_index+0.5)*self.block_meters
        # y = (y_index+0.5)*self.block_meters
        # x = x - self.offset
        # y = y - self.offset
        return math.inf, math.inf
    
    # def get_block_centers(self):
    #     """
    #     Get the centers of all blocks.
    #     :return: the centers of all blocks
    #     """
    #     centers = []
    #     for block in self.block_stack:
            
    #         centers.append(self.get_block_center(block))
    #     return numpy.array(centers)
    
    def get_block_centers(self):
        """
        Get the centers of all blocks.
        :return: the centers of all blocks
        """
        centers = []
        for block in self.block_stack:
            x, y = self.get_block_center(block)
            if (x,y)!=(math.inf, math.inf):
                centers.append((x,y))
        return numpy.array(centers)
    
    def get_coverage(self):
        """
        Get the coverage of the map.
        :return: the coverage of the map
        """
        return numpy.sum(self.visited == 2)/self.visited.size


