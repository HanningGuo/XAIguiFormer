import os
import torch
import numpy as np
import os.path as osp
from torch_geometric.data import InMemoryDataset, Data


class FreqbandData(Data):
    """
    A class to re-write several variable in Data in order to adapt to the model format
    """
    def __inc__(self, key, value, *args, **kwargs):
        if 'freqband' in key:
            return 1 + getattr(self, key)[-1]
        else:
            return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        return super().__cat_dim__(key, value, *args, **kwargs)


class EEGBenchmarkDataset(InMemoryDataset):
    """
    A variety of artificially graph, label and demographic information datasets

    :param root: str, root directory where the dataset should be saved
    :param name: str, The name of the dataset (one of TDBRAIN or TUAB)
    :param split: str (optional), If :obj:`"train"`, loads the training dataset.
            If :obj:`"val"`, loads the validation dataset.
            If :obj:`"test"`, loads the test dataset.
            (default: :obj:`"train"`)
    :param transform: callable (optional), A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
    :param pre_transform: callable (optional), A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    :param pre_filter: callable (optional), A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
    """

    names = ['TDBRAIN', 'TUAB']
    split_dataset = ['train', 'val', 'test']

    def __init__(self, root, name, split='train', transform=None, pre_transform=None, pre_filter=None):
        self.name = name
        assert self.name in self.names
        self.split = split
        assert self.split in self.split_dataset

        super().__init__(root, transform, pre_transform, pre_filter)

        path = self.processed_paths[self.split_dataset.index(self.split)]

        self.data, self.slices = torch.load(path)

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def raw_file_names(self):
        return [
            folder_name for folder_name in os.listdir(osp.join(self.raw_dir, self.split)) if os.path.isdir(osp.join(self.raw_dir, self.split, folder_name))
        ]

    @property
    def raw_paths(self):
        file_names = self.raw_file_names
        return [osp.join(self.raw_dir, self.split, file_name, f"{file_name}_EC_") for file_name in file_names]

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def processed_file_names(self):
        return ['train_dataset.pt', 'val_dataset.pt', 'test_dataset.pt']

    def download(self):
        pass

    def process(self):
        # load desired data and
        # convert them to the input format of Data object
        data_list = []
        for raw_path in self.raw_paths:
            cohs = np.load(raw_path + 'coherence.npy', allow_pickle=True)
            wplis = np.load(raw_path + 'wpli.npy', allow_pickle=True)
            label = np.load(raw_path + 'label.npy', allow_pickle=True)
            demographic_info = np.load(raw_path + 'demographics.npy', allow_pickle=True)
            eid = osp.split(raw_path)[1].split('_EC_')[0]

            for coh, wpli in zip(cohs, wplis):
                # create node features and corresponding order of sub-band
                coh = np.transpose(coh, (2, 0, 1))
                node_feat = coh.reshape((-1, coh.shape[2]))
                # create graph order
                freqband_order = torch.repeat_interleave(torch.arange(coh.shape[0]), repeats=coh.shape[1]).unsqueeze(1)

                # create edge index for each frequency band
                base_edge_index, edge_index_list = [], []
                for row in range(len(wpli)):
                    for col in range(row + 1, len(wpli)):
                        base_edge_index.append([row, col])
                base_edge_index = torch.tensor(base_edge_index, dtype=torch.long).t()
                for offset in range(wpli.shape[2]):
                    # add the offset for each sub-band
                    edge_index_list.append(base_edge_index + (offset * len(wpli)))

                # create edge features
                edge_attr = wpli[base_edge_index[0], base_edge_index[1], :].reshape((-1, 1), order='F')

                # create Data object using processed data
                data = FreqbandData(x=torch.from_numpy(node_feat).to(torch.float32),
                                    edge_index=torch.cat(edge_index_list, dim=1).contiguous(),
                                    edge_attr=torch.from_numpy(edge_attr).to(torch.float32),
                                    y=torch.from_numpy(label).to(torch.float32))
                # create frequency band order
                # convert to correct index by freqband_order % self.num_nodes
                data.freqband_order = freqband_order
                # add demographic information
                data.demographic_info = torch.from_numpy(demographic_info).t().to(torch.float32)
                # add EID to data
                data.eid = eid

                data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # save processed data to disk
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[self.split_dataset.index(self.split)])

    def __repr__(self):
        return f'{self.name}({len(self)})'
