import torch
from utils.abc_dataset import CustomDataset
import numpy
from collections.abc import Callable
from PIL import Image


class NumpyDataset(CustomDataset):
    def __init__(
        self,
        np_x: numpy.ndarray,
        np_y: numpy.ndarray,
        transform: Callable = None,
        feature_column_name: str = "raw_inputs",
        labels_column_name: str = "raw_labels",
        length_column_name: str = None,
    ):
        self.np_x = np_x
        self.np_y = np_y
        assert len(np_x) == len(np_y)
        self.transform = transform

        self.feature_column_name = feature_column_name
        self.labels_column_name = labels_column_name

        self.length_column_name = length_column_name

    def __len__(self):
        return len(self.np_x)

    def __getitem__(self, idx):
        if isinstance(idx, (int, slice)):
            xs = self.np_x[idx]
            ys = self.np_y[idx]
        elif isinstance(idx, (tuple, list)):
            xs = [self.np_x[i] for i in idx]
            ys = [self.np_y[i] for i in idx]

        if isinstance(idx, (slice, tuple, list)):
            for i in range(len(xs)):
                # set_transform effect
                if self.transform:
                    xs[i] = self.transform(xs[i])
                # items[i] = self.json_to_tensor(items[i])
        else:
            # set_transform effect
            if self.transform:
                xs[i] = self.transform(xs[i])
            # items = self.json_to_tensor(items)

        return {self.feature_column_name: xs, self.labels_column_name: ys}
