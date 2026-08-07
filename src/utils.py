import pandas as pd

def print_dataset_summary(participants):
    summary = []
    for participant_id, participant in participants.items():
        summary.append({
            "participant": participant_id,
            "train_trials": len(participant["train"]),
            "test_trials": len(participant["test"])
        })

    summary_df = pd.DataFrame(summary)
    print(summary_df.describe())
    return summary_df
