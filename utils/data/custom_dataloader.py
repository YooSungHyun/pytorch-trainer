import torch
from torch.nn.utils.rnn import pad_sequence


class CustomDataLoader(torch.utils.data.DataLoader):
    def __init__(self, feature_column_name: str = "inputs", labels_column_name="labels", *args, **kwargs):
        """
        Creates a data loader for AudioDatasets.
        """
        super(CustomDataLoader, self).__init__(*args, **kwargs)
        self.feature_column_name = feature_column_name
        self.labels_column_name = labels_column_name
        self.collate_fn = self._collate_fn

    def _pad(self, required_input, input_size: int, max_length: int = 0, padding_value: float = 0.0):
        difference = max_length - int(input_size)
        padded_output = torch.nn.functional.pad(required_input, (0, difference), "constant", value=padding_value)
        return padded_output

    def _collate_fn(self, batch):
        # make pad or something work for each step's batch
        inputs = list()
        labels = list()
        for i in range(len(batch)):
            inputs.append(torch.FloatTensor(batch[i][self.feature_column_name]))
            labels.append(torch.FloatTensor(batch[i][self.labels_column_name]))

        # if you neeed to many inputs, plz change this line
        # TODO(User): `inputs` must match the input argument of the model exactly (the current example only utilizes `inputs`).
        _returns = {"inputs": torch.stack(inputs), "labels": torch.stack(labels)}
        # _returns = {"input_ids", "attention_mask", "input_type_ids", "labels"}

        return _returns
