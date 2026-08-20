import numpy as np
from scipy.optimize import minimize

def predict_rt(value_difference, a, b, c):
    return a + b / (np.abs(value_difference) + c)

def rt_loss(params, df):
    a, b, c = params

    if a <= 0 or b <= 0 or c <= 0:
        return np.inf

    predicted = predict_rt(
        df["value_difference"],
        a,
        b,
        c,
    )

    return np.mean((df["RT"] - predicted) ** 2)

def fit_rt_model(df):

    df = df.dropna(subset=["RT"]).copy()

    result = minimize(
        rt_loss,
        x0=[1000, 3000, 1],
        args=(df,),
        bounds=[
            (100, 5000),
            (100, 10000),
            (0.001, 20),
        ],
    )

    return result

def add_predicted_rt(df, a, b, c):
    df = df.copy()

    df["predicted_RT"] = predict_rt(
        df["value_difference"],
        a,
        b,
        c
    )

    return df

def predict_rt_exponential(value_difference, a, b, c):
    return a + b * np.exp(-c * np.abs(value_difference))

def rt_loss_exponential(params, df):
    a, b, c = params

    if a <= 0 or b <= 0 or c <= 0:
        return np.inf

    predicted = predict_rt_exponential(
        df["value_difference"],
        a,
        b,
        c,
    )

    return np.mean((df["RT"] - predicted) ** 2)

def fit_rt_model_exponential(df):
    df = df.dropna(subset=["RT"]).copy()

    result = minimize(
        rt_loss_exponential,
        x0=[1000, 1000, 1],
        args=(df,),
        bounds=[
            (100, 5000),
            (100, 10000),
            (0.001, 20),
        ],
    )

    return result

def add_predicted_rt_exponential(df, a, b, c):
    df = df.copy()

    df["predicted_RT"] = predict_rt_exponential(
        df["value_difference"],
        a,
        b,
        c,
    )

    return df
