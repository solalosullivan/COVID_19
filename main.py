from model import EpidemicModel
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os

# TODO:change definition of these functions to clip values to [0,1]
def n(t):
    return 6e-3  # 6e-3


def eta(t):
    return 1  # 1 = no restriction


# At the beginning, 25 % of the population has been infected

parameters = {
    "n": n,
    "eta": eta,
    "gamma": 0.09,
    "N": 30,
    "M": 40,
    "R0_max": 15,
    "mu": 0.02,  # 0.02
    "C": 0.5,
    "sigma": 1,
    "h": 1e-6,
    "verbose": False,
}
t_max = 300

if __name__ == "__main__":

    model = EpidemicModel(**parameters)
    model.plot(t_max, animation=True)

    # save model
    with open(os.path.join("models", f"eta={eta(0)}__t_max={t_max}"), "wb") as f:
        pickle.dump(model, f)
