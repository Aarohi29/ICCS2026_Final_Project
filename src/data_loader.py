from pathlib import Path

import pandas as pd
from scipy.io import loadmat


COLUMNS = [
    "immOutcome",
    "delOutcome",
    "delay",
    "choice",
    "p_imm",
    "condition",
    "RT",
]


def load_participant(file_path):

    sample = loadmat(file_path)

    train_df = pd.DataFrame(
        sample["data_train"],
        columns=COLUMNS
    )

    test_df = pd.DataFrame(
        sample["data_test"],
        columns=COLUMNS
    )

    return train_df, test_df


def load_all_participants(data_dir):

    data_dir = Path(data_dir)

    participant_files = sorted(data_dir.glob("*.mat"))

    participants = {}

    for file in participant_files:

        train_df, test_df = load_participant(file)

        participants[file.stem] = {

            "train": train_df,

            "test": test_df

        }

    return participants
