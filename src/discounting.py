import numpy as np

def immediate_value(amount):
    return amount


def hyperbolic_value(amount, delay, k):
    return amount / (1 + k * delay)


def value_difference(immediate_amount, delayed_amount, delay, k):
    sv_immediate = immediate_value(immediate_amount)
    sv_delayed = hyperbolic_value(delayed_amount, delay, k)
    return sv_delayed - sv_immediate

def add_subjective_values(df, k):
    df = df.copy()
    df["SV_immediate"] = immediate_value(
        df["immOutcome"]
    )

    df["SV_delayed"] = hyperbolic_value(
        df["delOutcome"],
        df["delay"],
        k
    )

    df["value_difference"] = (
        df["SV_delayed"]
        - df["SV_immediate"]
    )
    return df
