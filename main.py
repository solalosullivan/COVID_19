from model import EpidemicModel
import matplotlib.pyplot as plt
import numpy as np

# TODO:change definition of these functions to clip values to [0,1]
def n(t):
    return 6e-3  # 6e-3


def eta(t):
    return 0.5


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
    "h": 1 / 1000,
    "verbose": False,
}
t_max = 20

if __name__ == "__main__":
    model = EpidemicModel(**parameters)
    model.plot(t_max, animation=True)
    plt.imshow(model.population.I[-1])
    plt.show()
