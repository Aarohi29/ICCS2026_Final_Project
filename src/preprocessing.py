import pandas as pd
def remove_missing_choices(df):
    return df[df["choice"] != 0].copy()

def remove_missing_rt(df):
    return df.dropna(subset=["RT"]).copy()

def reward_trials(df):
    return df[df["condition"] == 1].copy()

def loss_trials(df):
    return df[df["condition"] == 2].copy()

def get_split(participants, split="train"):
    return {
        pid: data[split]
        for pid, data in participants.items()
    }

def combine_participants(participant_dict):
    combined = []
    for participant_id, df in participant_dict.items():
        temp = df.copy()
        temp["participant"] = participant_id
        combined.append(temp)
    return pd.concat(combined, ignore_index=True)


