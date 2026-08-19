import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from src.discounting import add_subjective_values


def softmax_probability(value_difference, beta):
    return expit(beta*value_difference)


def negative_log_likelihood(params, df):
    k, beta = params

    if k <= 0 or beta <= 0:
        return np.inf

    df = add_subjective_values(df, k)

    probabilities = softmax_probability(
        df["value_difference"],
        beta
    )

    probabilities = np.clip(
        probabilities,
        1e-10,
        1 - 1e-10
    )

    choices = (df["choice"] == 2).astype(int)

    log_likelihood = (
        choices * np.log(probabilities)
        +
        (1 - choices) * np.log(1 - probabilities)
    )

    return -np.sum(log_likelihood)


def fit_choice_model(df):
    result = minimize(
        negative_log_likelihood,
        x0=[0.01, 1.0],
        args=(df,),
        bounds=[
            (1e-5, 2),
            (1e-5, 20)
        ]
    )

    return result


def predict_choices(df, k, beta):

    df = add_subjective_values(df, k)

    df["P_delayed"] = softmax_probability(
        df["value_difference"],
        beta
    )

    return df
