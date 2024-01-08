import os
from pathlib import Path
import torch
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    StateDictType,
    FullStateDictConfig,  # general model non-sharded, non-flattened params
    LocalStateDictConfig,  # flattened params, usable only by FSDP
    # ShardedStateDictConfig, # un-flattened param but shards, usable by other parallel schemes.
)

from torch.distributed._shard.checkpoint import FileSystemReader, FileSystemWriter, save_state_dict, load_state_dict


# create singleton saving policies to avoid making over and over
fullstate_save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)


def save_model_checkpoint(model, rank, checkpoint_folder, logger=None):
    """saving model via rank0 cpu streaming and full_state_dict"""

    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, fullstate_save_policy):
        cpu_state = model.state_dict()

    if logger:
        logger.info(f"saving process: rank {rank}  done w model state_dict\n")

    if rank == 0:
        if logger:
            logger.info("--> saving model ...")
        # create save path
        save_full_path = os.path.join(checkpoint_folder, "total_model.pt")

        # save model
        torch.save(cpu_state, save_full_path)

        if logger:
            logger.info(f"train end & model checkpoint saved at {save_full_path}\n")


def load_model_checkpoint(model, rank, checkpoint_filepath, logger=None):
    # where is the checkpoint at...
    full_state_dict_model_path = Path(checkpoint_filepath)
    # is it present...
    if not full_state_dict_model_path.is_file():
        if logger:
            logger.info(f"model checkpoint {full_state_dict_model_path} not present. Returning...")
        else:
            print(f"model checkpoint {full_state_dict_model_path} not present. Returning...")
        return

    model_checkpoint = torch.load(full_state_dict_model_path)
    # integrate into loaded model
    model.load_state_dict(model_checkpoint)

    if logger:
        logger.info(f"model checkpoint loaded to rank:{rank} {model.device.type}")


def load_distributed_model_checkpoint(model, rank, checkpoint_folder, logger=None):
    if logger:
        logger.info(f"loading distributed checkpoint, rank {rank}...")

    checkdir = Path(checkpoint_folder)

    if not checkdir.exists():
        if rank == 0:
            if logger:
                logger.info("No checkpoint directory found...skipping")
        return

    reader = FileSystemReader(checkdir)

    with FSDP.state_dict_type(model, StateDictType.LOCAL_STATE_DICT):
        state_dict = model.state_dict()
        load_state_dict(state_dict, reader)
        model.load_state_dict(state_dict)

    if logger:
        logger.info(f"--> local state loaded on rank {rank}")

    return


def save_distributed_model_checkpoint(model, checkpoint_folder):
    # distributed checkpoint saving

    # create writer to current path
    save_dir = Path(checkpoint_folder)

    writer = FileSystemWriter(save_dir)

    with FSDP.state_dict_type(model, StateDictType.LOCAL_STATE_DICT):
        state_dict = model.state_dict()

    # write out distributed checkpoint
    save_state_dict(state_dict, writer)

    return


def load_client_state(state: dict, checkpoint_filepath: str, device: str = "cpu", logger=None):
    """Load model checkpoint.

    Args:
        checkpoint_filepath: checkpoint filename including path.
        model: torch model.
        optimizer: torch optimizer.

    Returns:
        state: dict
    """
    assert os.path.isfile(checkpoint_filepath)
    checkpoint_dict = torch.load(checkpoint_filepath, map_location=device)

    # epoch
    current_epoch = (
        checkpoint_dict["current_epoch"]
        if ("current_epoch" in checkpoint_dict.keys() and checkpoint_dict["current_epoch"] is not None)
        else 0
    )
    state.update({"current_epoch": current_epoch})

    # global_step
    global_step = (
        checkpoint_dict["global_step"]
        if "global_step" in checkpoint_dict.keys() and checkpoint_dict["global_step"] is not None
        else 0
    )
    state.update({"global_step": global_step})

    # step
    step = checkpoint_dict["step"] if "step" in checkpoint_dict.keys() and checkpoint_dict["step"] is not None else 0
    state.update({"step": step})

    # dtype
    dtype = (
        checkpoint_dict["dtype"]
        if "dtype" in checkpoint_dict.keys() and checkpoint_dict["dtype"] is not None
        else torch.float32
    )
    state.update({"dtype": dtype})

    # optimizer
    if (
        "optimizer" in state.keys()
        and state["optimizer"] is not None
        and "optimizer" in checkpoint_dict.keys()
        and checkpoint_dict["optimizer"] is not None
    ):
        state["optimizer"].load_state_dict(checkpoint_dict["optimizer"])

    # scheduler_cfg
    if (
        "scheduler_cfg" in state.keys()
        and state["scheduler_cfg"] is not None
        and "scheduler" in checkpoint_dict.keys()
        and checkpoint_dict["scheduler"] is not None
    ):
        state["scheduler_cfg"]["scheduler"].load_state_dict(checkpoint_dict["scheduler"])

    # trainable_loss
    if (
        "grad_scaler" in state.keys()
        and state["grad_scaler"] is not None
        and "grad_scaler" in checkpoint_dict.keys()
        and checkpoint_dict["grad_scaler"] is not None
    ):
        state.update({"grad_scaler": checkpoint_dict["grad_scaler"]})

    # trainable_loss
    if (
        "trainable_loss" in state.keys()
        and state["trainable_loss"] is not None
        and "trainable_loss" in checkpoint_dict.keys()
        and checkpoint_dict["trainable_loss"] is not None
    ):
        state["trainable_loss"].load_state_dict(checkpoint_dict["trainable_loss"], strict=False)

    if logger is not None:
        logger.info(f"Loaded client_state '{checkpoint_filepath}' epoch: {current_epoch} global_step: {global_step}")
