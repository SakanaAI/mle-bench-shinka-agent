# EVOLVE-BLOCK-START
import os
from typing import Literal

# NOTE: DATA_DIR is read-only, please write only either to AGENT_DIR or RESULTS_DIR
# NOTE: COPYING answer from `validation_answer` IS NOT ALLOWED AND IS CONSIDERED CHEATING
# NOTE: The training time is limited to *one hour*.
# NOTE: The data exists in the DATA_DIR directory. Read the files from the directory. DO NOT BUILD A SYNTHETIC DATASET.

# these globals must not be changed
DATA_DIR = os.environ.get("DATA_DIR")
AGENT_DIR = os.environ.get("AGENT_DIR")
RESULTS_DIR = os.environ.get("RESULTS_DIR")


class Model:
    # An example model class
    # Could be simply replaced by a sklearn, Torch or Transformers model
    def __init__(self, seed: int | None = None):
        self.seed = seed
        self.prepare_data()

    def train(self, *args, **kwargs):
        """
        Training function

        It should produce *two* machine learning models.
        One trained exclusively on the "train" split,
        while the other is trained on *both* "train" and "validation" splits.

        The train-only model (let's name it `self.model["val"]`) should be used for reporting
        the performance on the validation split (`self.make_submission(split="validation")`).

        The other model (let's name it `self.model["test"]`) should be used for
        submitting the final predictions (self.make_submission(split="test")).

        This way we can maximize the final performance while avoiding data contamination
        in the validation step.

        It is a good idea to handle a potential class imbalance issue.
        """
        pass

    def make_submission(self, split: Literal["validation", "test"]):
        """
        Make a submission file for a given split.

        Args:
            split (Literal["validation", "test"]): which split to make submission for

        Returns:
            str: path to the submission file
        """
        submission_path = f"{RESULTS_DIR}/submission_{split}.csv"
        # An example code that submits the example submission
        # In practice, we should load the test file, make predictions, and save the predictions as the submission file
        # The submission file must follow the format described in the task description.
        # The submission file must be saved at {RESULTS_DIR}/submission_{split}.csv
        #
        # IMPORTANT: You should make sure that this function works as the model will be evaluated based on the submission file
        # Leaving the code as is will result in an error in the evaluation
        # Cheating by copying labels from existing validation answer is NOT allowed
        #
        # e.g., this could be something like
        # data = self.load_data(split)
        # pred = self.predict(data, model=split) # the model argument selects which model to use
        # formatted_submission = self.format_submission(data, pred)
        # formatted_submission.to_csv(submission_path, index=False)
        pass

        return submission_path

    def prepare_data(self):
        # Load data based on the task description from DATA_DIR
        # e.g., `train.zip` mentioned in the description can be accessed at f"{DATA_DIR}/train.zip"
        #
        # Depending on the competition, the `train` file could be a csv or a zip or other file formats
        # and it could contain the features (e.g., tabular tasks) or image ids (e.g., image classification tasks) or something else entirely.
        # You have to write code that handles the data for the current competition correctly.
        #
        # Do an 80/20 split seeded by self.seed on the training file to get the train and validation splits
        # We should try to make sure that the performance on the validation split is correlated with the unseen test split and that we can detect when the model is overfitted to the train set
        #
        # You can write the train split wherever you want, but make sure that `self.load_data` can read it
        # Write the validation split into two files: {AGENT_DIR}/validation.csv and {AGENT_DIR}/validation_answer.csv
        # The answer file should have the same format as the submission example described in the task description.
        # The format typically contains some kind of id (e.g., "Id" column) and prediction (e.g., "is_positive" column).
        #
        # IMPORTANT: The exact location is important. Make sure that the files are saved exactly as specified.
        # Leaving the code as is will result in an error in the evaluation
        pass

    def load_data(self, split: Literal["train", "validation", "test"]):
        # load the corresponding data split
        pass


def train_model(seed: int | None = None):
    """
    Train a machine learning model on training data

    The trained model will be evaluated on different data splits

    This function must return a functional `model` instance.

    Returns:
        model: a trained machine learning model
    """
    # Example code
    model = Model(seed=seed)
    model.train()
    return model


# EVOLVE-BLOCK-END

from utils import TrainingTimeoutError, training_timeout


def run_mle_bench(*args, **kwargs):
    import traceback

    print("Start training model...")
    try:
        with training_timeout(3600):  # one hour
            return train_model(kwargs.get("seed", None))
    except TrainingTimeoutError as exc:
        raise RuntimeError(
            "train_model exceeded the maximum runtime of 1 hour."
        ) from exc
    except Exception:
        raise RuntimeError(
            f"train_model failed with the following error trace: {traceback.format_exc()}"
        )
