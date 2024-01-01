import torch
from utils.abc_dataset import CustomDataset
import pandas
from collections.abc import Callable
from PIL import Image


class PandasDataset(CustomDataset):
    def __init__(
        self,
        dataframe: pandas,
        seq_length: int,
        transform: Callable = None,
        img_col: str = None,
        length_column_name: str = None,
    ):
        self.jsonl = dataframe.to_dict("records")
        self.transform = transform
        self.img_col = img_col
        self.length_column_name = length_column_name
        self.seq_length = seq_length

    def __len__(self):
        return len(self.jsonl)

    # def json_to_tensor(self, obj):
    #     if isinstance(obj, dict):  # dict obj
    #         return {k: self.json_to_tensor(v) for k, v in obj.items()}
    #     elif isinstance(obj, list):  # list obj
    #         return torch.tensor([self.json_to_tensor(item) for item in obj])
    #     else:  # just type obj
    #         return torch.tensor(obj)

    def __getitem__(self, idx):
        if isinstance(idx, (int, slice)):
            items = self.jsonl[idx]
        elif isinstance(idx, (tuple, list)):
            items = [self.jsonl[i] for i in idx]

        if isinstance(idx, (slice, tuple, list)):
            for i in range(len(items)):
                if self.img_col and self.img_col in items[i].keys():
                    img_name = items[i][self.img_col]
                    image = Image.open(img_name)
                    items[i].update({"image": image})
                # set_transform effect
                if self.transform:
                    items[i] = self.transform(items[i])
                # you must get length after transform
                if self.length_column_name:
                    items[i].update({"lengths": len(items[i][self.length_column_name])})
                # items[i] = self.json_to_tensor(items[i])
        else:
            if self.img_col and self.img_col in items.keys():
                img_name = items[self.img_col]
                image = Image.open(img_name)
                items.update({"image": image})
            # set_transform effect
            if self.transform:
                items[i] = self.transform(items[i])
            # you must get length after transform
            if self.length_column_name:
                items[i].update({"lengths": len(items[i][self.length_column_name])})
            # items = self.json_to_tensor(items)

        return items
