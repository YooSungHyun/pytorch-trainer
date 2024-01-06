from dataclasses import dataclass
import simple_parsing as sp


@dataclass
class TrainingArguments:
    """Help string for this group of command-line arguments"""

    train_datasets_path: str = None
    eval_datasets_path: str = None
    metric_on_cpu: bool = False
    transformers_model_name: str = None
    train_data_ratio: float = 0.7
    seed: int = None  # all seed
    local_rank: int = None  # ddp local rank
    data_dir: str = "datasets"  # target pytorch lightning data dirs
    output_dir: str = "model_outputs"  # model output path
    num_workers: int = 1  # how many proc map?
    optim_beta1: float = 0.9
    optim_beta2: float = 0.999
    optim_eps: float = 1e-08
    learning_rate: float = 0.001  # learning rate (if warmup, this is max_lr)
    warmup_ratio: float = 0.2  # learning rate scheduler warmup ratio per EPOCH
    div_factor: int = 25  # initial_lr = max_lr/div_factor
    final_div_factor: int = 1e4  # (max_lr/div_factor)*final_div_factor is final lr
    weight_decay: float = 0.0001  # weigth decay
    per_device_train_batch_size: int = 1  # The batch size per GPU/TPU core/CPU for training.
    per_device_eval_batch_size: int = 1  # The batch size per GPU/TPU core/CPU for evaluation.
    dropout_p: float = 0.0  # Drop path rate (default: 0.0)
    cutoff_epoch: int = 0  # if drop_mode is early / late, this is the epoch where dropout ends / starts
    drop_mode: str = sp.field(default="standard", choices=["standard", "early", "late"])  # drop mode
    drop_schedule: str = sp.field(
        default="constant", choices=["constant", "linear"]
    )  # drop schedule for early dropout / s.d. only
    group_by_length: bool = False
    accumulate_grad_batches: int = 1
    max_epochs: int = 1
    log_every_n: int = 1
    length_column_name: str = None
    feature_column_name: str = "raw_inputs"
    labels_column_name: str = "raw_labels"
    dataloader_drop_last: bool = False
    sampler_shuffle: bool = True
    max_norm: float = 0.0  # gradient clipping max_norm value
    model_dtype: str = "fp32"
    fsdp_config: str = "./fsdp_config/config.json"
