import torch
from torch.nn.utils.rnn import pad_sequence


class CustomDataLoader(torch.utils.data.DataLoader):
    def __init__(self, key_to_inputs=["inputs"], key_to_labels=["labels"], *args, **kwargs):
        """
        Creates a data loader for AudioDatasets.
        """
        super(CustomDataLoader, self).__init__(*args, **kwargs)
        self.key_to_inputs = key_to_inputs
        self.key_to_labels = key_to_labels
        self.collate_fn = self._collate_fn

    def _collate_fn(self, batch):
        # make pad or something work for each step's batch
        inputs = list()
        labels = list()
        for i in range(len(batch)):
            for key in self.key_to_inputs:
                inputs.append(torch.FloatTensor(batch[i][key]))
            for key in self.key_to_labels:
                labels.append(torch.FloatTensor(batch[i][key]))

        return torch.stack(inputs), torch.stack(labels)
