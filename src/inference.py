import pandas as pd

from src.choice_model import fit_choice_model


def fit_all_participants(participants, split="train"):
    results = []

    for participant_id, participant_data in participants.items():

        df = participant_data[split]

        result = fit_choice_model(df)

        results.append({

            "participant": participant_id,
            "k": result.x[0],
            "beta": result.x[1],
            "negative_log_likelihood": result.fun,
            "success": result.success

        })

    return pd.DataFrame(results)
