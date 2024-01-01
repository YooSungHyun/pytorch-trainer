import dataclasses
import os
import random
from collections import OrderedDict, defaultdict
from copy import deepcopy
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch


def dataclass_to_namespace(args, args_name):
    # Dataclass arg to python namespace
    if args.__contains__(args_name):
        for key, value in args.__getattribute__(args_name).__dict__.items():
            args.__setattr__(key, value)
        args.__delattr__(args_name)
    return args


def seed_everything(random_seed: int) -> None:
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


# Copyright The PyTorch Lightning team.
# Licensed under the Apache License, Version 2.0 (the "License");
#     http://www.apache.org/licenses/LICENSE-2.0
#


def is_namedtuple(obj: object) -> bool:
    """Check if object is type nametuple."""
    # https://github.com/pytorch/pytorch/blob/v1.8.1/torch/nn/parallel/scatter_gather.py#L4-L8
    return isinstance(obj, tuple) and hasattr(obj, "_asdict") and hasattr(obj, "_fields")


def is_dataclass_instance(obj: object) -> bool:
    """Check if object is dataclass."""
    # https://docs.python.org/3/library/dataclasses.html#module-level-decorators-classes-and-functions
    return dataclasses.is_dataclass(obj) and not isinstance(obj, type)


def apply_to_collection(
    data: Any,
    dtype: Union[type, Any, Tuple[Union[type, Any]]],
    function: Callable,
    *args: Any,
    wrong_dtype: Optional[Union[type, Tuple[type, ...]]] = None,
    include_none: bool = True,
    allow_frozen: bool = False,
    **kwargs: Any,
) -> Any:
    """Recursively applies a function to all elements of a certain dtype.

    Args:
        data: the collection to apply the function to
        dtype: the given function will be applied to all elements of this dtype
        function: the function to apply
        *args: positional arguments (will be forwarded to calls of ``function``)
        wrong_dtype: the given function won't be applied if this type is specified and the given collections
            is of the ``wrong_dtype`` even if it is of type ``dtype``
        include_none: Whether to include an element if the output of ``function`` is ``None``.
        allow_frozen: Whether not to error upon encountering a frozen dataclass instance.
        **kwargs: keyword arguments (will be forwarded to calls of ``function``)

    Returns:
        The resulting collection

    """
    if include_none is False or wrong_dtype is not None or allow_frozen is True:
        # not worth implementing these on the fast path: go with the slower option
        return _apply_to_collection_slow(
            data,
            dtype,
            function,
            *args,
            wrong_dtype=wrong_dtype,
            include_none=include_none,
            allow_frozen=allow_frozen,
            **kwargs,
        )
    # fast path for the most common cases:
    if isinstance(data, dtype):  # single element
        return function(data, *args, **kwargs)
    if isinstance(data, list) and all(isinstance(x, dtype) for x in data):  # 1d homogeneous list
        return [function(x, *args, **kwargs) for x in data]
    if isinstance(data, tuple) and all(isinstance(x, dtype) for x in data):  # 1d homogeneous tuple
        return tuple(function(x, *args, **kwargs) for x in data)
    if isinstance(data, dict) and all(isinstance(x, dtype) for x in data.values()):  # 1d homogeneous dict
        return {k: function(v, *args, **kwargs) for k, v in data.items()}
    # slow path for everything else
    return _apply_to_collection_slow(
        data,
        dtype,
        function,
        *args,
        wrong_dtype=wrong_dtype,
        include_none=include_none,
        allow_frozen=allow_frozen,
        **kwargs,
    )


def _apply_to_collection_slow(
    data: Any,
    dtype: Union[type, Any, Tuple[Union[type, Any]]],
    function: Callable,
    *args: Any,
    wrong_dtype: Optional[Union[type, Tuple[type, ...]]] = None,
    include_none: bool = True,
    allow_frozen: bool = False,
    **kwargs: Any,
) -> Any:
    # Breaking condition
    if isinstance(data, dtype) and (wrong_dtype is None or not isinstance(data, wrong_dtype)):
        return function(data, *args, **kwargs)

    elem_type = type(data)

    # Recursively apply to collection items
    if isinstance(data, Mapping):
        out = []
        for k, v in data.items():
            v = _apply_to_collection_slow(
                v,
                dtype,
                function,
                *args,
                wrong_dtype=wrong_dtype,
                include_none=include_none,
                allow_frozen=allow_frozen,
                **kwargs,
            )
            if include_none or v is not None:
                out.append((k, v))
        if isinstance(data, defaultdict):
            return elem_type(data.default_factory, OrderedDict(out))
        return elem_type(OrderedDict(out))

    is_namedtuple_ = is_namedtuple(data)
    is_sequence = isinstance(data, Sequence) and not isinstance(data, str)
    if is_namedtuple_ or is_sequence:
        out = []
        for d in data:
            v = _apply_to_collection_slow(
                d,
                dtype,
                function,
                *args,
                wrong_dtype=wrong_dtype,
                include_none=include_none,
                allow_frozen=allow_frozen,
                **kwargs,
            )
            if include_none or v is not None:
                out.append(v)
        return elem_type(*out) if is_namedtuple_ else elem_type(out)

    if is_dataclass_instance(data):
        # make a deepcopy of the data,
        # but do not deepcopy mapped fields since the computation would
        # be wasted on values that likely get immediately overwritten
        fields = {}
        memo = {}
        for field in dataclasses.fields(data):
            field_value = getattr(data, field.name)
            fields[field.name] = (field_value, field.init)
            memo[id(field_value)] = field_value
        result = deepcopy(data, memo=memo)
        # apply function to each field
        for field_name, (field_value, field_init) in fields.items():
            v = None
            if field_init:
                v = _apply_to_collection_slow(
                    field_value,
                    dtype,
                    function,
                    *args,
                    wrong_dtype=wrong_dtype,
                    include_none=include_none,
                    allow_frozen=allow_frozen,
                    **kwargs,
                )
            if not field_init or (not include_none and v is None):  # retain old value
                v = getattr(data, field_name)
            try:
                setattr(result, field_name, v)
            except dataclasses.FrozenInstanceError as e:
                if allow_frozen:
                    # Quit early if we encounter a frozen data class; return `result` as is.
                    break
                raise ValueError(
                    "A frozen dataclass was passed to `apply_to_collection` but this is not allowed."
                ) from e
        return result

    # data is neither of dtype, nor a collection
    return data


def apply_to_collections(
    data1: Optional[Any],
    data2: Optional[Any],
    dtype: Union[type, Any, Tuple[Union[type, Any]]],
    function: Callable,
    *args: Any,
    wrong_dtype: Optional[Union[type, Tuple[type]]] = None,
    **kwargs: Any,
) -> Any:
    """Zips two collections and applies a function to their items of a certain dtype.

    Args:
        data1: The first collection
        data2: The second collection
        dtype: the given function will be applied to all elements of this dtype
        function: the function to apply
        *args: positional arguments (will be forwarded to calls of ``function``)
        wrong_dtype: the given function won't be applied if this type is specified and the given collections
            is of the ``wrong_dtype`` even if it is of type ``dtype``
        **kwargs: keyword arguments (will be forwarded to calls of ``function``)

    Returns:
        The resulting collection

    Raises:
        AssertionError:
            If sequence collections have different data sizes.

    """
    if data1 is None:
        if data2 is None:
            return None
        # in case they were passed reversed
        data1, data2 = data2, None

    elem_type = type(data1)

    if isinstance(data1, dtype) and data2 is not None and (wrong_dtype is None or not isinstance(data1, wrong_dtype)):
        return function(data1, data2, *args, **kwargs)

    if isinstance(data1, Mapping) and data2 is not None:
        # use union because we want to fail if a key does not exist in both
        zipped = {k: (data1[k], data2[k]) for k in data1.keys() | data2.keys()}
        return elem_type(
            {
                k: apply_to_collections(*v, dtype, function, *args, wrong_dtype=wrong_dtype, **kwargs)
                for k, v in zipped.items()
            }
        )

    is_namedtuple_ = is_namedtuple(data1)
    is_sequence = isinstance(data1, Sequence) and not isinstance(data1, str)
    if (is_namedtuple_ or is_sequence) and data2 is not None:
        if len(data1) != len(data2):
            raise ValueError("Sequence collections have different sizes.")
        out = [
            apply_to_collections(v1, v2, dtype, function, *args, wrong_dtype=wrong_dtype, **kwargs)
            for v1, v2 in zip(data1, data2)
        ]
        return elem_type(*out) if is_namedtuple_ else elem_type(out)

    if is_dataclass_instance(data1) and data2 is not None:
        if not is_dataclass_instance(data2):
            raise TypeError(
                "Expected inputs to be dataclasses of the same type or to have identical fields"
                f" but got input 1 of type {type(data1)} and input 2 of type {type(data2)}."
            )
        if not (
            len(dataclasses.fields(data1)) == len(dataclasses.fields(data2))
            and all(map(lambda f1, f2: isinstance(f1, type(f2)), dataclasses.fields(data1), dataclasses.fields(data2)))
        ):
            raise TypeError("Dataclasses fields do not match.")
        # make a deepcopy of the data,
        # but do not deepcopy mapped fields since the computation would
        # be wasted on values that likely get immediately overwritten
        data = [data1, data2]
        fields: List[dict] = [{}, {}]
        memo: dict = {}
        for i in range(len(data)):
            for field in dataclasses.fields(data[i]):
                field_value = getattr(data[i], field.name)
                fields[i][field.name] = (field_value, field.init)
                if i == 0:
                    memo[id(field_value)] = field_value

        result = deepcopy(data1, memo=memo)

        # apply function to each field
        for (field_name, (field_value1, field_init1)), (_, (field_value2, field_init2)) in zip(
            fields[0].items(), fields[1].items()
        ):
            v = None
            if field_init1 and field_init2:
                v = apply_to_collections(
                    field_value1,
                    field_value2,
                    dtype,
                    function,
                    *args,
                    wrong_dtype=wrong_dtype,
                    **kwargs,
                )
            if not field_init1 or not field_init2 or v is None:  # retain old value
                return apply_to_collection(data1, dtype, function, *args, wrong_dtype=wrong_dtype, **kwargs)
            try:
                setattr(result, field_name, v)
            except dataclasses.FrozenInstanceError as e:
                raise ValueError(
                    "A frozen dataclass was passed to `apply_to_collections` but this is not allowed."
                ) from e
        return result

    return apply_to_collection(data1, dtype, function, *args, wrong_dtype=wrong_dtype, **kwargs)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim,
    scheduler_cfg: dict,
    current_epoch: int,
    global_step: int,
    step: int,
    checkpoint_filepath: str,
    trainable_loss=None,
    logger=None,
):
    """Save model epoch, global step, state dict and optimizer.

    Args:
        model: torch model.
        optimizer: torch optimizer.
        epoch: epoch.
        global_step: global step.
        checkpoint_filepath: checkpoint filename including path.
    """

    if logger is not None:
        logger.info("Saving model and optimizer state at epoch {} to {}".format(current_epoch, checkpoint_filepath))
    if hasattr(model, "module"):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()

    optimizer_state_dict = None
    if optimizer is not None:
        optimizer_state_dict = optimizer.state_dict()

    scheduler_state_dict = None
    if scheduler_cfg["scheduler"] is not None:
        scheduler_state_dict = scheduler_cfg["scheduler"].state_dict()

    loss_state_dict = None
    if trainable_loss is not None:
        if hasattr(trainable_loss, "module"):
            loss_state_dict = trainable_loss.module.state_dict()
        else:
            loss_state_dict = trainable_loss.state_dict()

    torch.save(
        {
            "current_epoch": current_epoch,
            "global_step": global_step,
            "step": step,
            "model": state_dict,
            "optimizer": optimizer_state_dict,
            "scheduler": scheduler_state_dict,
            "trainable_loss": loss_state_dict,
        },
        checkpoint_filepath,
    )


def load_checkpoint(
    state: dict,
    checkpoint_filepath: str,
    device: str = "cpu",
    logger=None,
    load_bias: bool = True,
):
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
        "trainable_loss" in state.keys()
        and state["trainable_loss"] is not None
        and "trainable_loss" in checkpoint_dict.keys()
        and checkpoint_dict["trainable_loss"] is not None
    ):
        state["trainable_loss"].load_state_dict(checkpoint_dict["trainable_loss"], strict=False)

    # model
    assert "model" in checkpoint_dict, "checkpoint_dict must have model state dict."
    saved_state_dict = checkpoint_dict["model"]
    # 불러온 체크포인트와의 shape등을 비교하여, 맞지 않으면 기존에 초기화된걸 쓰기 위한 로직
    # saved_state_dict은 불러올 모델, state_dict 초기화된 모델을 의미한다.
    if hasattr(state["model"], "module"):
        state_dict = state["model"].module.state_dict()
    else:
        state_dict = state["model"].state_dict()

    # 먼가 다르면 초기화된 weight를 쓰고, 아니면 불러온다.
    new_state_dict = {}
    for k in state_dict.keys():
        if k in saved_state_dict.keys() and state_dict[k].size() == saved_state_dict[k].size():
            if not load_bias and k.split(".")[-1] == "bias":
                new_state_dict[k] = state_dict[k]
            else:
                new_state_dict[k] = saved_state_dict[k]

    if hasattr(state["model"], "module"):
        state["model"].module.load_state_dict(new_state_dict)
    else:
        state["model"].load_state_dict(new_state_dict)

    if logger is not None:
        logger.info(f"Loaded checkpoint '{checkpoint_filepath}' epoch: {current_epoch} global_step: {global_step}")


def tensor_dict_to_device(tensor_dict: Dict[str, torch.Tensor], device: str = "cpu", non_blocking: bool = False):
    assert isinstance(tensor_dict, dict), f"tensor_dict is not dicts. Found {type(tensor_dict)}."

    for k, v in tensor_dict.items():
        if isinstance(v, dict):
            tensor_dict_to_device(v, device, non_blocking=non_blocking)
        elif isinstance(v, torch.Tensor):
            tensor_dict[k] = v.to(device, non_blocking=non_blocking)
        else:
            raise TypeError(f"value of dict is not torch.Tensor. Found {type(v)}")
