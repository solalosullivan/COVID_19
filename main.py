from model import EpidemicModel

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
    "sigma": 1 / 1000,  # 1 / 1000,
    "h": 1 / 1000,
    "verbose": False,
}
t_max = 100

if __name__ == "__main__":
    model = EpidemicModel(**parameters)
    model.plot(t_max, animation=True)
