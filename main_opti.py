import random as rd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

from model import EpidemicModel, Parameters
from utils import f_of_L


def simulated_annealing(
    parameters: Parameters, T_i: float, T_f: float, c: float
) -> Tuple[List[float], List[float], float]:
    """Optimization using simulated annealing

    Args:
        parameters (Parameters): model parameters
        T_i (float): initial temperature
        T_f (float): final temperature
        c (float): cooling rate
    """

    # Initial control: random piecewise constant functions
    n_k = [rd.choice(n_vals) for _ in range(nb_per)]
    eta_k = [rd.choice(eta_vals) for _ in range(nb_per)]

    # Set functions for model
    parameters["n"] = f_of_L(n_k, t_per)
    parameters["eta"] = f_of_L(eta_k, t_per)

    # Run model
    model_k = EpidemicModel(**parameters)
    for t in range(t_max - 1):
        model_k.iter()
    D_k = model_k.population.D[
        -1
    ]  # Decision criterion (score): final death toll

    # Initializing the best control encountered
    n_opt = n_k
    eta_opt = eta_k
    D_opt = D_k

    # Initializing the temperature of the system
    T = T_i
    D_list = []

    it = 0
    nb_it = int(np.log(T_f / T_i) / np.log(c))
    while T > T_f:
        if it % 10 == 0:
            print("---------------------------------------")
            print(f"iteration {it} out of {nb_it} iterations")
        # Randomly changing random values of n_k and eta_k (one each)
        # Picking the values which change
        i_n = rd.randint(0, nb_per - 1)
        i_eta = rd.randint(0, nb_per - 1)
        # Assigning new values
        n_new = list(n_k)
        eta_new = list(eta_k)
        n_new[i_n] = rd.choice(n_vals)
        eta_new[i_eta] = rd.choice(eta_vals)

        # Run model again
        parameters["n"] = f_of_L(n_new, t_per)
        parameters["eta"] = f_of_L(eta_new, t_per)
        model_new = EpidemicModel(**parameters)
        for t in range(t_max - 1):
            model_new.iter()
        D_new = model_new.population.D[-1]

        # Computing the acceptance probability
        proba = np.exp(
            -(D_new - D_k) / T
        )  # proba > 1 if the new control is better

        if (
            rd.random() <= proba
        ):  # The new control is adopted if it is better or was (randomly) accepted
            D_k = D_new
            n_k = n_new
            eta_k = eta_new
            D_list.append(D_new)

        # Updating the optimal value
        if D_new < D_opt:
            D_opt = D_new
            n_opt = n_new
            eta_opt = eta_new

        # Cooling the system
        T *= c
        it = it + 1

    fig, ax1 = plt.subplots()

    # Run model
    parameters["n"] = f_of_L(n_opt, t_per)
    parameters["eta"] = f_of_L(eta_opt, t_per)
    model_opt = EpidemicModel(**parameters)
    model_opt.plot(t_max=t_max, animation=True)

    # plot the first curve on the left axis
    ax1.step(
        [k * t_per for k in range(nb_per)],
        n_opt,
        label=r"n(t)",
        color="pink",
    )
    ax1.set_xlabel("Time")
    ax1.set_ylabel(r"n(t)", color="pink")
    ax1.legend(loc="lower right")

    # create a second axis on the right
    ax2 = ax1.twinx()

    # plot the second curve on the right axis
    ax2.step(
        [k * t_per for k in range(nb_per)],
        eta_opt,
        label=r"$\eta(t)$",
        color="purple",
    )
    ax2.set_ylabel(r"$\eta(t)$", color="purple")
    ax2.legend()

    plt.show()

    return n_opt, eta_opt, D_opt


def genetic(parameters, nb_choices, nb_iter, p_breed, p_mut):
    # nb_choices is the number of controls considered at once -- it is a multple of 4.
    # nb_iter is the number of iterations of the algorithm
    # p_breed is the breeding probability
    # p_mut is the probability of mutation

    # Initializing the possible controls randomly
    controls = []
    for i in range(nb_choices):
        n = [rd.choice(n_vals) for _ in range(nb_per)]
        eta = [rd.choice(eta_vals) for _ in range(nb_per)]
        parameters["n"] = f_of_L(n, t_per)
        parameters["eta"] = f_of_L(eta, t_per)
        model = EpidemicModel(**parameters)
        for t in range(t_max - 1):
            model.iter()
        D = model.population.D[-1]
        controls.append((n, eta, D))

    # Initializing the best value encountered
    # Extracting the scores
    D_list = [l[2] for l in controls]
    # Localizing the minimal score
    ind = D_list.index(min(D_list))
    # Extracting the associated control
    best = controls[ind]
    model_opt = None

    for i in range(nb_iter):
        if i % 10 == 0:
            print("------------------------------------------")
            print(f"iteration {i} out of {nb_iter}")

        # Selecting nb_choices/2 controls depending on their score
        D_list = [
            1 - controls[k][2] for k in range(nb_choices)
        ]  # The weights are the survival rates
        selected = rd.choices(controls, weights=D_list, k=nb_choices // 2)
        controls = list(selected)  # Removing the other controls

        # Creation of nb_choices/2 new controls from the selected ones
        for j in range(nb_choices // 4):
            control_1, control_2 = rd.sample(
                selected, 2
            )  # Choosing two controls among the selected ones
            (n_1, eta_1, D_1) = control_1
            (n_2, eta_2, D_2) = control_2

            if rd.random() <= p_breed:  # Breeding the selected controls
                # Choosing where to cut
                ind_n_cut = rd.randint(0, nb_per - 1)
                ind_eta_cut = rd.randint(0, nb_per - 1)
                # Creating breeded controls
                n_temp = n_1[: (ind_n_cut + 1)] + n_2[(ind_n_cut + 1) :]
                n_2 = n_2[: (ind_n_cut + 1)] + n_1[(ind_n_cut + 1) :]
                n_1 = list(n_temp)

                eta_temp = (
                    eta_1[: (ind_eta_cut + 1)] + eta_2[(ind_eta_cut + 1) :]
                )
                eta_2 = eta_2[: (ind_eta_cut + 1)] + eta_1[(ind_eta_cut + 1) :]
                eta_1 = list(eta_temp)

            if rd.random() <= p_mut:  # Mutation of the controls
                # Picking the values which mutate
                ind_n_1_mut = rd.randint(0, nb_per - 1)
                ind_n_2_mut = rd.randint(0, nb_per - 1)
                ind_eta_1_mut = rd.randint(0, nb_per - 1)
                ind_eta_2_mut = rd.randint(0, nb_per - 1)
                # Assigning new values
                n_1[ind_n_1_mut] = rd.choice(n_vals)
                n_2[ind_n_2_mut] = rd.choice(n_vals)
                eta_1[ind_eta_1_mut] = rd.choice(eta_vals)
                eta_2[ind_eta_2_mut] = rd.choice(eta_vals)

            # Adding the new controls to the list and updating the best value
            parameters["n"] = f_of_L(n_1, t_per)
            parameters["eta"] = f_of_L(eta_1, t_per)
            model_1 = EpidemicModel(**parameters)
            for t in range(t_max - 1):
                model_1.iter()
            D_1 = model_1.population.D[-1]
            controls.append((n_1, eta_1, D_1))
            if D_1 < best[2]:
                best = (n_1, eta_1, D_1)

            parameters["n"] = f_of_L(n_2, t_per)
            parameters["eta"] = f_of_L(eta_2, t_per)
            model_2 = EpidemicModel(**parameters)
            for t in range(t_max - 1):
                model_2.iter()
            D_2 = model_2.population.D[-1]
            controls.append((n_2, eta_2, D_2))
            if D_2 < best[2]:
                best = (n_2, eta_2, D_2)

    (n_opt, eta_opt, D_opt) = best

    # Run model
    parameters["n"] = f_of_L(n_opt, t_per)
    parameters["eta"] = f_of_L(eta_opt, t_per)
    model_opt = EpidemicModel(**parameters)
    model_opt.plot(t_max=t_max, animation=True)

    # Display optimal control
    fig, ax1 = plt.subplots()

    # plot the first curve on the left axis
    ax1.step(
        [k * t_per for k in range(nb_per)],
        n_opt,
        label=r"n(t)",
        color="pink",
    )
    ax1.set_xlabel("Time")
    ax1.set_ylabel(r"n(t)", color="pink")
    ax1.legend(loc="lower right")

    # create a second axis on the right
    ax2 = ax1.twinx()

    # plot the second curve on the right axis
    ax2.step(
        [k * t_per for k in range(nb_per)],
        eta_opt,
        label=r"$\eta(t)$",
        color="purple",
    )
    ax2.set_ylabel(r"$\eta(t)$", color="purple")
    ax2.legend()

    plt.show()

    return best


if __name__ == "__main__":
    nb_n = 10  # number of possible values for n
    nb_eta = 10  # number of possible values for eta
    n_sup = 0.01  # n_max
    eta_inf = 0.2  # eta_inf
    t_per = 10  # time of eac period (every 10 days)
    nb_per = 6  # number of period
    t_max = t_per * nb_per

    parameters = {
        "n": None,
        "eta": None,
        "gamma": 0.09,
        "N": 10,  # 30 but very long with real value
        "M": 15,  # 40 but very long with real value
        "R0_max": 15,
        "mu": 0.02,  # 0.02
        "C": 0.5,
        "sigma": 1,
        "h": 1e-6,
        "verbose": False,
    }

    n_vals = np.linspace(
        0, n_sup, nb_n
    )  # Discretization of the vaccination rate
    eta_vals = np.linspace(eta_inf, 1, nb_eta)  # Discretization of the NPI

    method = "genetic"  # TODO: choose method here

    if method == "annealing":
        T_i = 1000  # initial temperature
        T_f = 1  # final temperature
        c = 0.99  # cooling rate
        n_opt, eta_opt, D_opt = simulated_annealing(parameters, T_i, T_f, c)

    if method == "genetic":
        nb_choices = 8  # nb_choices is the number of controls considered at once -- it is a multple of 4.
        nb_iter = 50  # nb_iter is the number of iterations of the algorithm
        p_breed = 0.8  # p_breed is the breeding probability
        p_mut = 0.05  # p_mut is the probability of mutation
        n_opt, eta_opt, D_opt = genetic(
            parameters, nb_choices, nb_iter, p_breed, p_mut
        )
