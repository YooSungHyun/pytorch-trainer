from dataclasses import dataclass


@dataclass
class InferenceArguments:
    """Help string for this group of command-line arguments"""

    seed: int = 42  # all seed
    transformers_model_name: str = None
    local_rank: int = 0  # ddp local rank
    data_path: str = None
    model_path: str = "model_outputs"  # target pytorch lightning model dir
    per_device_test_batch_size: int = 1  # The batch size per GPU/TPU core/CPU for evaluation.
    num_workers: int = 1
    metric_on_cpu: bool = False  # If you want to run validation_step on cpu -> true
    model_dtype: str = "fp32"
    feature_column_name: str = "input_ids"
    labels_column_name: str = "labels"
    dataloader_drop_last: bool = False
    sampler_shuffle: bool = True
    length_column_name: str = None
    group_by_length: bool = False
    fsdp_config: str = "./fsdp_config/config.json"
    accumulate_grad_batches: int = 1
