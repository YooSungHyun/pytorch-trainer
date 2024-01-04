import os
from pathlib import Path
from datetime import datetime
import torch
import time

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    StateDictType,
    FullStateDictConfig,  # general model non-sharded, non-flattened params
    LocalStateDictConfig,  # flattened params, usable only by FSDP
    # ShardedStateDictConfig, # un-flattened param but shards, usable by other parallel schemes.
)

from torch.distributed._shard.checkpoint import (
    FileSystemReader,
    FileSystemWriter,
    save_state_dict,
    load_state_dict,
)
from torch.distributed.checkpoint.default_planner import (
    DefaultSavePlanner,
    DefaultLoadPlanner,
)


# from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType
import torch.distributed._shard.checkpoint as dist_cp
import torch.distributed as dist


# create singleton saving policies to avoid making over and over
fullstate_save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)


def load_model_sharded(model, rank, checkpoint_folder, logger=None):
    # torch.manual_seed(103)

    reader = FileSystemReader(Path(checkpoint_folder))

    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
        checkpoint = model.state_dict()
        if rank == 0:
            ck = checkpoint.keys()
            logger.info(f" checkpoint key len = {len(ck)} and \n keys =  {ck}")

        dist_cp.load_state_dict(
            state_dict=checkpoint,
            storage_reader=reader,
        )
        if rank == 0:
            logger.info("checkpoint after load_state_dict()")
            ck = checkpoint.keys()
            logger.info(f" checkpoint key len = {len(ck)} and \n keys =  {ck}")
        model.load_state_dict(checkpoint)
    if rank == 0:
        logger.info(f"Sharded state checkpoint loaded from {checkpoint_folder}")


def save_model_and_optimizer_sharded(model, rank, checkpoint_folder, optim=None, logger=None):
    """save model and optimizer via sharded_state_dict to save_dir"""

    if rank == 0:
        print(f"Saving model to {checkpoint_folder}")

    distributed_writer = dist_cp.FileSystemWriter(checkpoint_folder)

    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
        state_dict = {"model": model.state_dict()}
        if optim is not None:
            state_dict["optim"] = FSDP.optim_state_dict(model, optim)

        dist_cp.save_state_dict(
            state_dict=state_dict,
            storage_writer=distributed_writer,
            planner=DefaultSavePlanner(),
        )
    dist.barrier()
    if rank == 0:
        logger.info(f"Sharded state checkpoint saved to {checkpoint_folder}")


def save_model_checkpoint(model, optimizer, rank, checkpoint_folder, checkpoint_type, epoch=1, logger=None):
    """saving model via rank0 cpu streaming and full_state_dict"""

    # saving with rank0 cpu
    if not checkpoint_type == StateDictType.FULL_STATE_DICT:
        logger.info(f" unable to handle checkpoint type {checkpoint_type}, aborting")

    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, fullstate_save_policy):
        cpu_state = model.state_dict()

    logger.info(f"saving process: rank {rank}  done w model state_dict\n")

    if rank == 0:
        logger.info("--> saving model ...")
        # create save path
        save_full_path = os.path.join(checkpoint_folder, f"epoch-{epoch:04d}.ckpt")

        # save model
        torch.save(cpu_state, save_full_path)

        logger.info(f"model checkpoint saved for epoch {epoch} at {save_full_path}\n")


def load_model_checkpoint(model, rank, checkpoint_filepath, logger=None):
    """load local checkpoint to rank0 cpu
    must be called * before * passing to FSDP"""

    if rank != 0:
        return

    # where is the checkpoint at...
    full_state_dict_model_path = Path(checkpoint_filepath)
    # is it present...
    if not full_state_dict_model_path.is_file():
        print(f"model checkpoint {full_state_dict_model_path} not present. Returning...")
        return

    model_checkpoint = torch.load(full_state_dict_model_path)
    # integrate into loaded model
    model.load_state_dict(model_checkpoint)

    logger.info("model checkpoint loaded to rank0 cpu")


def save_optimizer_checkpoint(model, optimizer, rank, checkpoint_folder, epoch=1, logger=None):
    """save optimizer state via full state dict"""

    logger.info("--> optim state call on rank {rank}\n")

    # pull all sharded optimizer states to rank0 cpu...

    optim_state = FSDP.full_optim_state_dict(model, optimizer)

    logger.info("optim state dict ready on {rank} and len of {len(optim_state)}\n")

    if rank == 0:
        opt_save_full_path = os.path.join(checkpoint_folder, "optimizer.pt")
        logger.info("--> saving optimizer state...")

        torch.save(optim_state, opt_save_full_path)
        logger.info(f"--> saved {opt_save_full_path} to disk")


def load_optimizer_checkpoint(model, optimizer, rank, optimizer_ckpt_filepath, logger=None):
    """load an fdsp optimizer full_state checkpoint using scatter method
    this ensures only rank 0 loads the optimizer state dict and scatters to other ranks
    """

    opt_file_path = Path(optimizer_ckpt_filepath)

    if not opt_file_path.is_file():
        logger.info(f"warning - optimizer checkpoint not present {opt_file_path}. Returning. ")
        return

    full_osd = None

    if rank == 0:
        full_osd = torch.load(opt_file_path)
        logger.info("loaded full osd on rank 0")

    # called from all ranks, though only rank0 has a valid param for full_osd
    sharded_osd = FSDP.scatter_full_optim_state_dict(full_osd, model)

    logger.info(f"optimizer shard loaded on rank {rank}")


def load_distributed_model_checkpoint(model, rank, checkpoint_type, checkpoint_folder, logger=None):
    if checkpoint_type == StateDictType.LOCAL_STATE_DICT:
        logger.info(f"loading distributed checkpoint, rank {rank}...")

        checkdir = Path(checkpoint_folder)

        if not checkdir.exists():
            if rank == 0:
                logger.info("No checkpoint directory found...skipping")
            return

        reader = FileSystemReader(checkdir)

        with FSDP.state_dict_type(
            model,
            StateDictType.LOCAL_STATE_DICT,
        ):
            state_dict = model.state_dict()
            load_state_dict(state_dict, reader)
            model.load_state_dict(state_dict)

        logger.info(f"--> local state loaded on rank {rank}")

        return


def save_distributed_model_checkpoint(model, rank, checkpoint_type, checkpoint_folder, epoch=1):
    # distributed checkpoint saving

    # confirm type of checkpoint and save
    if checkpoint_type == StateDictType.LOCAL_STATE_DICT:
        # create writer to current path
        save_dir = Path(checkpoint_folder)

        writer = FileSystemWriter(
            save_dir,
        )

        with FSDP.state_dict_type(
            model,
            StateDictType.LOCAL_STATE_DICT,
        ):
            state_dict = model.state_dict()

        # write out distributed checkpoint
        save_state_dict(state_dict, writer)

        return
