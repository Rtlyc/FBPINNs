# import numpy as np 
import torch


class FastTensorDataLoader:
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).
    Source: https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6
    """
    def __init__(self, *tensors, batch_size=32, shuffle=False):
        """
        Initialize a FastTensorDataLoader.
        :param *tensors: tensors to store. Must have the same length @ dim 0.
        :param batch_size: batch size to load.
        :param shuffle: if True, shuffle the data *in-place* whenever an
            iterator is created out of this object.
        :returns: A FastTensorDataLoader.
        """
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors

        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches
    def __iter__(self):
        if self.shuffle:
            r = torch.randperm(self.dataset_len)
            self.tensors = [t[r] for t in self.tensors]
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        batch = tuple(t[self.i:self.i+self.batch_size] for t in self.tensors)
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches

def generate_sphere_data(limit=1, radius=0.5, minimum=0.1, maximum=1, sample_size=1000):
    #! return points, speeds, bounds
    allpoints = [] # n*2
    allspeeds = [] # 
    allbounds = []
    wholesize = 0
    while wholesize < sample_size:
        start_points = torch.rand(sample_size*8, 2)*2 - 1
        # end_points = torch.rand(sample_size*8, 2)*2 - 1
        # generate uniform distance from start_points
        direction = torch.rand(sample_size*8, 2)*2 - 1
        direction = direction / torch.sqrt(torch.sum(direction**2, axis=1, keepdim=True))
        delta_length = torch.rand(sample_size*8, 1) * 1.0
        end_points = start_points + direction * delta_length
        start_points *= limit
        end_points *= limit 


        sdf_s = torch.sqrt(torch.sum(start_points**2, axis=1)) - radius
        sdf_e = torch.sqrt(torch.sum(end_points**2, axis=1)) - radius
        valid = (sdf_s>=0) & (sdf_e>=0) & (sdf_s<maximum)
        sdf_s = sdf_s[valid, None]
        sdf_e = sdf_e[valid, None]
        start_points = start_points[valid]
        end_points = end_points[valid]

        points = torch.cat((start_points, end_points), dim=1)
        bounds = torch.cat((sdf_s, sdf_e), dim=1)
        speeds = torch.clip(bounds, minimum, maximum)/maximum

        allpoints.append(points)
        allbounds.append(bounds)
        allspeeds.append(speeds)
        wholesize += len(points)

    allpoints = torch.cat(allpoints, dim=0)[:sample_size]
    allbounds = torch.cat(allbounds, dim=0)[:sample_size]
    allspeeds = torch.cat(allspeeds, dim=0)[:sample_size]

    return allpoints, allspeeds, allbounds 

if __name__ == "__main__":
    points, speeds, bounds = generate_sphere_data()
    print(points.shape)
    print(speeds.shape)
    print(bounds.shape)
    





    