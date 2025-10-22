import os

import pandas as pd


def find_label_col():
    data_dir = os.environ.get("DATA_DIR", "/home/data")
    df_train = pd.read_csv(f"{data_dir}/train.csv", nrows=0)
    df_test = pd.read_csv(f"{data_dir}/test.csv", nrows=0)
    label_cols = set(df_train.columns) - set(df_test.columns)
    assert len(label_cols) == 1, (
        f"Expected exactly one label column, but found: {label_cols}"
    )
    return list(label_cols)[0]


def cross_validation_split(seed: int | None = None):
    data_dir = os.environ.get("DATA_DIR", "/home/data/")

    train_path = os.path.join(data_dir, "train.csv")
    df = pd.read_csv(train_path)

    train_df = df.sample(frac=0.8, random_state=seed)
    val_df = df.drop(train_df.index)

    # write train.csv with the 80% split and write validation.csv with the rest
    agent_dir = os.environ.get("AGENT_DIR", "/home/agent/")
    train_df.to_csv(os.path.join(agent_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(agent_dir, "validation.csv"), index=False)
