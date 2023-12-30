from abc import abstractmethod
import torch


class CustomDataset(torch.utils.data.Dataset):
    @abstractmethod
    def __init__(self):
        pass

    # 총 데이터의 개수를 리턴
    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, idx):
        pass
