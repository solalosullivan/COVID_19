import pickle
import os
import matplotlib.pyplot as plt
import json
from datetime import datetime

from model import EpidemicModel, Parameters


def n(
    t: int,
    t_half_pop_vaccinated: int,
    max_vacc_rate: float,
    max_pop_vaccinated_per_day: float,
    remaining_population: int,
) -> float:
    """Vaccination rate, parabolic but clipped at max_vacc_rate"""
    alpha = 3 / (
        t_half_pop_vaccinated
        * (2 * t_half_pop_vaccinated + 1)
        * (t_half_pop_vaccinated + 1)
    )

    return max(
        min(
            max_vacc_rate,
            alpha * t**2,
            max_pop_vaccinated_per_day * remaining_population,
        ),
        0,
    )  


def eta(t: int) -> float:  # noqa
    """Restriction (1 means no restriction, 0 means full restriction)"""
    # This function can depend on time but it is constant here
    return 0.4


# Model parameters
parameters = Parameters(
    n=n,
    eta=eta,
    gamma=0.09,
    N=30,
    M=40,
    R0_max=15,
    mu=0.02,
    C=0.5,
    sigma=1,
    h=1e-6,
    verbose=True,
    inf_prop=0.0001,
    rec_prop=0,
    t_half_pop_vaccinated=200,  # time obective to reach half of the population vaccinated (objective because the rate is bounded)
    max_vacc_rate=1
    / 300,  # maximum 1/300 of the total population vaccinated per day
    max_pop_vaccinated_per_day=1
    / 100,  # maximum 1/10 of the remaining population vaccinated per day (smooth the tail of curve)
)


# Model duration (days) parameter
t_max = 301

if __name__ == "__main__":

    model = EpidemicModel(parameters=parameters)
    fig = model.plot(t_max=t_max, animation=True)

    # save model
    model_name = "model_" + datetime.now().strftime("%m|%d_%H:%M:%S")
    model_path = os.path.join("models", model_name)
    os.makedirs(model_path)
    print(f"{model_path} folder created.")
    with open(os.path.join(model_path, model_name + ".pkl"), "wb") as f:
        pickle.dump(model, f)
    print(f"Model : {model_name}.pkl saved.")

    # save parameters
    with open(os.path.join(model_path, "parameters.json"), "w") as json_file:
        parameters_dict = parameters.__dict__
        parameters_dict.pop("n")
        parameters_dict.pop("eta")
        json.dump(parameters_dict, json_file, indent=4)

    print(f"Parameters : parameters.json saved.")

    # save population plot
    fig.savefig(
        os.path.join(model_path, f"{model_name}_population_plot.png"),
        dpi=300,
        bbox_inches="tight",
    )
    print(f"Population plot : {model_name}_population_plot.png saved.")

    # save variants evolution plot
    for j in range(0, t_max, t_max // 5):
        variants_frame = model.population.I[j]
        t = model.T[j]
        plt.imshow(variants_frame, cmap="viridis", interpolation="nearest")
        cbar = plt.colorbar()
        cbar.set_label("Proportion of population hit by variant")
        plt.xlabel("Resistance to vaccine (%)")
        plt.ylabel("Transmissibility")
        plt.title(f"Variants distribution at time {t}")
        plt.gcf().patch.set_facecolor("white")
        plt.savefig(
            os.path.join("models", model_name, f"virus_variants_time_{t}.png"),
            dpi=300,
            bbox_inches="tight",
            transparent=False,
        )
        plt.close()
        if j == 0:
            print("Virus variants propagation :")
        print(f"virus_variants_time_{t}.png saved.")
