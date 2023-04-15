import random as rd
from model import Population, EpidemicModel
import numpy as np
import matplotlib.pyplot as plt

nb_n = 10
nb_eta = 10
n_sup = 0.001
eta_inf = 0.1
t_max = 40

n_vals = np.linspace(0, n_sup, nb_n)  # Discretization of the vaccination rate
eta_vals = np.linspace(eta_inf, 1, nb_eta)  # Discretization of the NPI


def n_L(L):
    def n_t(t):
        return L[int(t)]

    return n_t


def eta_L(L):
    def eta_t(t):
        return L[int(t)]

    return eta_t


def annealing(T_i, T_f, c):
    # T_i is the initial temperature
    # T_f is the final temperature
    # c is the cooling rate
    parameters = {
        "n": None,
        "eta": None,
        "gamma": 0.09,
        "N": 3,
        "M": 3,
        "R0_max": 15,
        "mu": 0.02,  # 0.02
        "C": 0.5,
        "sigma": 1 / 1000,  # 1 / 1000,
        "h": 1 / 1000,
        "verbose": False,
    }

    # Initial control: maximal vaccine rate and minimal NPI
    n_k = [rd.choice(n_vals) for _ in range(t_max)]
    eta_k = [rd.choice(eta_vals) for _ in range(t_max)]
    parameters["n"] = n_L(n_k)
    parameters["eta"] = eta_L(eta_k)
    model_k = EpidemicModel(**parameters)
    for t in range(t_max - 1):
        model_k.iter()
    D_k = model_k.population.D[-1]  # Decision criterion (score): final death toll

    # Initializing the best control encountered
    n_opt = n_k
    eta_opt = eta_k
    D_opt = D_k

    # Initializing the temperature of the system
    T = T_i
    D_list = []

    while T > T_f:
        # Randomly changing random values of n_k and eta_k (one each)
        # Picking the values which change
        i_n = rd.randint(0, nb_n - 1)
        i_eta = rd.randint(0, nb_eta - 1)
        # Assigning new values
        n_new = list(n_k)
        eta_new = list(eta_k)
        n_new[i_n] = rd.choice(n_vals)
        eta_new[i_eta] = rd.choice(eta_vals)

        # Computing the new score
        parameters["n"] = n_L(n_new)
        parameters["eta"] = eta_L(eta_new)
        model_k = EpidemicModel(**parameters)
        for t in range(t_max - 1):
            model_k.iter()
        D_new = model_k.population.D[-1]

        # Computing the acceptance probability
        proba = np.exp(-(D_new - D_k) / T)  # proba > 1 if the new control is better

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

    plt.plot(D_list)
    plt.show()
    print(f"D_opt={D_opt}")
    print("eta", eta_opt)
    print("n", n_opt)

    plt.plot(model_k.T, model_k.population.S, label="Susceptible", color="blue")
    plt.plot(model_k.T, model_k.population.D, label="Dead", color="black")
    plt.plot(model_k.T, model_k.population.V, label="Vaccinated", color="g")
    plt.plot(
        model_k.T,
        [np.sum(frame) for frame in model_k.population.I],
        label="Infected",
        color="red",
    )
    plt.plot(
        model_k.T,
        [np.sum(frame) for frame in model_k.population.R],
        label="Recovered",
        color="yellow",
    )
    plt.step(
        model_k.T,
        n_opt,
        label="N",
        color="pink",
    )
    plt.step(
        model_k.T,
        eta_opt,
        label="Eta",
        color="purple",
    )
    plt.xlabel("Time")
    plt.ylabel("Proportion of population")
    plt.legend()
    plt.show()

    return n_opt, eta_opt, D_opt


annealing(1000, 1, 0.99)


# def genetique(nb_choices, nb_iter, p_breed, p_mut) :
#     # nb_choices is the number of controls considered at once -- it is an even number.
#     # nb_iter is the number of iterations of the algorithm
#     # p_breed is the breeding probability
#     # p_mut is the probability of mutation

#     # Initializing the possible controls randomly
#     controls = []
#     for i in range(nb_choices) :
#         n = [rd.choice(n_vals) for _ in range(t_max)]
#         eta = [rd.choice(eta_vals) for _ in range(t_max)]
#         parameters.n = n
#         parameters.eta = eta
#         model = EpidemicModel((**parameters))
#         D = model.D[-1]
#         controls.append((n, eta, D))

#     # Initializing the best value encountered
#         # Extracting the scores
#     D_list = controls[:][2]
#         # Localizing the minimal score
#     ind = D_list.index(min(D_list))
#         # Extracting the associated control
#     best = controls[ind]

#     for i in range(nb_iter) :

#         # Selecting nb_choices/2 controls depending on their score
#         D_list = [1 - controls[k][2] for k in range(nb_choices)]    # The weights are the survival rates
#         selected = rd.choices(controls, weights = D_list, k = nb_choices // 2)
#         controls = list(selected)   # Removing the other controls

#         # Creation of nb_choices/2 new controls from the selected ones
#         for j in range(nb_choices//2) :
#             control_1, control_2 = rd.sample(selected, 2)   # Choosing two controls among the selected ones
#             (n_1, eta_1, D_1) = control_1
#             (n_2, eta_2, D_2) = control_2

#             if rd.random() <= p_breed :      # Breeding the selected controls
#                     # Choosing where to cut
#                 ind_n_cut = rd.randint(0, nb_n - 1)
#                 ind_eta_cut = rd.randint(0, nb_eta - 1)
#                     # Creating breeded controls
#                 n_temp = n_1[: (ind_n_cut + 1)] + n_2[(ind_n_cut + 1):]
#                 n_2 = n_2[: (ind_n_cut + 1)] + n_1[(ind_n_cut + 1):]
#                 n_1 = list(n_temp)

#                 eta_temp = eta_1[: (ind_eta_cut + 1)] + eta_2[(ind_eta_cut + 1):]
#                 eta_2 = eta_2[: (ind_eta_cut + 1)] + eta_1[(ind_eta_cut + 1):]
#                 eta_1 = list(eta_temp)

#             if rd.random() <= p_mut :       # Mutation of the controls
#                     # Picking the values which mutate
#                 ind_n_1_mut = rd.randint(0, nb_n - 1)
#                 ind_n_2_mut = rd.randint(0, nb_n - 1)
#                 ind_eta_1_mut = rd.randint(0, nb_eta - 1)
#                 ind_eta_2_mut = rd.randint(0, nb_eta - 1)
#                     # Assigning new values
#                 n_1[ind_n_1_mut] = rd.choice(n_vals)
#                 n_2[ind_n_2_mut] = rd.choice(n_vals)
#                 eta_1[ind_eta_1_mut] = rd.choice(eta_vals)
#                 eta_2[ind_eta_2_mut] = rd.choice(eta_vals)

#             # Adding the new controls to the list and updating the best value
#             parameters.n = n_1
#             parameters.eta = eta_1
#             model_1 = EpidemicModel((**parameters))
#             D_1 = model_1.population.D[-1]
#             controls.append((n_1, eta_1, D_1))
#             if D_1 < best[2] :
#                 best = (n_1, eta_1, D_1)

#             parameters.n = n_2
#             parameters.eta = eta_2
#             model_2 = EpidemicModel((**parameters))
#             D_2 = model_2.population.D[-1]
#             controls.append((n_2, eta_2, D_2))
#             if D_2 < best[2] :
#                 best = (n_2, eta_2, D_2)

#     return best
