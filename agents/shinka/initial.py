# EVOLVE-BLOCK-START
import os
import random
from typing import Literal

import pandas as pd

# NOTE: DATA_DIR is read-only, please write only either to AGENT_DIR or RESULTS_DIR

# these globals must not be changed
DATA_DIR = os.environ.get("DATA_DIR")
AGENT_DIR = os.environ.get("AGENT_DIR")
RESULTS_DIR = os.environ.get("RESULTS_DIR")


class Model:
    # An example model class
    # Could be simply replaced by a sklearn or Torch model
    def __init__(self, seed: int | None = None):
        self.seed = seed
        self.prepare_data()

    def predict(self, x):
        # An example predict class
        if isinstance(x, (pd.DataFrame, pd.Series)):
            n = len(x)
            # Return a Series aligned with the input index for convenience
            return pd.Series([random.randint(0, 1) for _ in range(n)], index=x.index)

        # Generic list/array-like batch (but not strings/bytes)
        try:
            if not isinstance(x, (str, bytes)) and len(x) > 1:
                return [random.randint(0, 1) for _ in range(len(x))]
        except TypeError:
            pass

        return random.randint(0, 1)

    def train(self, *args, **kwargs):
        pass

    def make_submission(self, split: Literal["validation", "test"]):
        # An example code that submits the example submission
        # In practice, we should load the test file, make predictions, and save the predicions as the submission file
        # The submission file should follow the format described in the task description.
        # The submission file should be saved at {RESULTS_DIR}/submission.csv
        #
        # IMPORTANT: You should make sure that this function works as the model will be evaluated based on the submission file
        # Leaving the code as is will result in an error in the evaluation
        # Cheating by copying labels from the validation data is not allowed
        if split == "test":
            cmd = f"cp {AGENT_DIR}/sample_submission.csv {RESULTS_DIR}/submission_test.csv"
            os.system(cmd)
            
            # e.g., this could be something like
            # data_test = self.load_data("test")
            # pred_test = self.predict(data_test)
            # self.format_submission(data_test, pred_test)
        elif split == "validation":
            # e.g., this could be something like
            # data_val = self.load_data("validation")
            # pred_val = self.predict(data_val)
            # self.format_submission(data_val, pred_val)
            raise NotImplementedError("Validation submission not implemented yet.")
        return f"{RESULTS_DIR}/submission_{split}.csv"

    def prepare_data(self):
        # Load data based on the task description from DATA_DIR
        # e.g., `train.zip` mentioned in the description can be accessed at f"{DATA_DIR}/train.zip"
        #
        # Depending on the competition, the `train` file could be a csv or a zip or other file formats
        # and it could contain the features (e.g., tabular tasks) or image ids (e.g., image classification task).
        # You have to write code that handle the data for the current competition correctly.
        #
        # Do a 80/20 split seeded by self.seed on the training file to get the train and validation splits
        # You can write the train split wherever you want, but make sure that `self.load_data` can read it
        # Write the validation split into two files: {AGENT_DIR}/validation.csv and {AGENT_DIR}/validation_answer.csv
        # The answer file should have the same format as the submission example shown in the task description.
        # The format typically contains some kind of id (e.g., "Id" column) and prediction (e.g., "is_positive" column).
        #
        # IMPORTANT: The exact location is important. Make sure that the files are saved exactly as specified.
        # Leaving the code as is will result in an error in the evaluation
        pass

    def load_data(self, split: Literal["train", "validation", "test"]):
        # load the corresponding data split
        return None


def train_model(seed: int | None = None):
    """
    Train a machine learning model on training data

    The trained model will be evaluated on different data splits

    Returns:
        model: a trained machine learning model
    """
    # Example code
    model = Model(seed=seed)
    model.train(model.load_data("train"))
    return model


# EVOLVE-BLOCK-END


def run_mle_bench(*args, **kwargs):
    # from utils import cross_validation_split, find_label_col

    # print("Start spliting data...")
    # cross_validation_split(kwargs.get("seed"))
    # agent_dir = os.environ.get("AGENT_DIR", "/home/agent/")
    # train_data = pd.read_csv(f"{agent_dir}/train.csv")

    # label_col = find_label_col()
    # x_train = train_data.drop(columns=[label_col])
    # y_train = train_data[label_col]
    # print(f"Feature columns: {x_train.columns}")
    # print(f"Label column: {label_col}")

    print("Start training model...")
    return train_model(kwargs.get("seed", None))
