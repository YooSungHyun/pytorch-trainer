import logging
from logging import StreamHandler

import numpy as np
import pandas as pd
import torch
from arguments.inference_args import InferenceArguments
from networks.models import Net
from setproctitle import setproctitle
from simple_parsing import ArgumentParser
from sklearn.preprocessing import MinMaxScaler
from utils.comfy import dataclass_to_namespace, seed_everything
from utils.model_checkpointing.common_handler import load_checkpoint

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(levelname)8s] %(message)s")

timeFileHandler = StreamHandler()
timeFileHandler.setFormatter(formatter)

logger.addHandler(timeFileHandler)


def main(hparams: InferenceArguments):
    # reference: https://www.kaggle.com/code/anitarostami/lstm-multivariate-forecasting
    setproctitle("cpu_inference")
    seed_everything(hparams.seed)

    # I'm not saved MinMaxScaler, so, have to re-calculate, stupid thing...🤣
    df_train = pd.read_csv("./raw_data/LSTM-Multivariate_pollution.csv", header=0, encoding="utf-8")
    # Kaggle author Test Final RMSE: 0.06539
    df_eval = pd.read_csv(hparams.data_path, header=0, encoding="utf-8")

    df_train_scaled = df_train.copy()
    df_test_scaled = df_eval.copy()

    # Define the mapping dictionary
    mapping = {"NE": 0, "SE": 1, "NW": 2, "cv": 3}

    # Replace the string values with numerical values
    df_train_scaled["wnd_dir"] = df_train_scaled["wnd_dir"].map(mapping)
    df_test_scaled["wnd_dir"] = df_test_scaled["wnd_dir"].map(mapping)

    df_train_scaled["date"] = pd.to_datetime(df_train_scaled["date"])
    # Resetting the index
    df_train_scaled.set_index("date", inplace=True)
    logger.info(df_train_scaled.head())

    scaler = MinMaxScaler()

    # Define the columns to scale
    columns = ["pollution", "dew", "temp", "press", "wnd_dir", "wnd_spd", "snow", "rain"]

    df_test_scaled = df_test_scaled[columns]

    # Scale the selected columns to the range 0-1
    df_train_scaled[columns] = scaler.fit_transform(df_train_scaled[columns])
    df_test_scaled[columns] = scaler.transform(df_test_scaled[columns])

    # we don't need to df_train_scaled anymore

    # Show the scaled data
    logger.info(df_test_scaled.head())

    df_test_scaled = np.array(df_test_scaled)

    n_future = 1
    n_past = 11
    #  Test Sets
    x = []
    y = []
    for i in range(n_past, len(df_test_scaled) - n_future + 1):
        x.append(df_test_scaled[i - n_past : i, 1 : df_test_scaled.shape[1]])
        y.append(df_test_scaled[i + n_future - 1 : i + n_future, 0])
    x_test, y_test = np.array(x), np.array(y)

    logger.info("X_test shape : {}      y_test shape : {} ".format(x_test.shape, y_test.shape))

    precision_dict = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
    mixed_precision = False
    if hparams.model_dtype in ["fp16", "float16"]:
        hparams.model_dtype = precision_dict["fp16"]
        mixed_precision = True
    elif hparams.model_dtype in ["bf16" or "bfloat16"]:
        hparams.model_dtype = precision_dict["bf16"]
        mixed_precision = True
    else:
        hparams.model_dtype = precision_dict["fp32"]

    model = Net()
    state = {"model": model}
    load_checkpoint(state, hparams.model_path)
    state["model"].eval()
    torch.set_grad_enabled(False)
    inputs = torch.tensor(np.array(x_test), dtype=hparams.model_dtype, device="cpu")

    outputs = model(inputs=inputs)
    mse = torch.nn.MSELoss()
    mse_loss = mse(outputs, torch.tensor(np.array(y_test), dtype=hparams.model_dtype, device="cpu"))
    rmse = torch.sqrt(mse_loss)
    logger.info(f"RMSE Loss is {rmse:0.10f}")
    np_outputs = np.concatenate([outputs, df_test_scaled[n_past:, 1:]], axis=1)
    np_labels = df_test_scaled[n_past:]
    np_outputs = scaler.inverse_transform(np_outputs)
    np_labels = scaler.inverse_transform(np_labels)

    result = np.concatenate([np_outputs[:, [0]], np_labels[:, [0]]], axis=1)
    pd_result = pd.DataFrame(result, columns=["pred", "labels"])
    pd_result.to_excel("./cpu_result.xlsx", index=False)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(InferenceArguments, dest="training_args")
    args = parser.parse_args()
    args = dataclass_to_namespace(args, "training_args")

    main(args)
