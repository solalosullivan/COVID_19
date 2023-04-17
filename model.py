import numpy as np
from utils import dotdot, permut, psi
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import Callable


class Population:
    def __init__(self, N: int, M: int, I0: np.ndarray, R0: np.ndarray):
        # S(t) = percentage of the population, which is susceptible to be infected, but has not
        # been infected so far
        # I[i,j](t) denotes the number of people infected by variant with transmissibility beta_i and
        # resistance omega_j
        # V(t) denotes the part of population vaccinated at time t
        # R[i,j](t) denotes the percentage of population that has recovered from variant (i,j)
        # D(t) denotes the percentage of population that has died
        self.S = [1 - np.sum(I0) - np.sum(R0)]
        self.I = [I0]
        self.V = [0]
        self.R = [R0]
        self.D = [0]
        self.r = [np.sum(self.R[0])]  # sum of recovered
        self.i = [np.sum(self.I[0])]  # sum of infected


class EpidemicModel:
    def __init__(
        self,
        n: Callable[[float], float],
        eta: Callable[[float], float],
        gamma: float,
        N: int,
        M: int,
        R0_max: float,
        mu: float,
        sigma: float,
        C: float,
        h: float,
        verbose: bool,
        inf_prop: float = 200e-5,
        rec_prop: float = 25 / 100,
    ):
        self.N = N  # transmissibility step
        self.M = M  # vaccine resistance step
        self.population = Population(
            N,
            M,
            self.init_I(i0=5, j0=1, prop=inf_prop),
            self.init_R(i0=5, j0=1, prop=rec_prop),
        )
        self.n = n  # vaccine rate
        self.eta = eta  # level of liberty
        self.beta = (
            R0_max * gamma * np.array([j / (N - 1) for j in range(N)]).reshape((-1, 1))
        )  # transmission rates
        self.omega = np.array([j / (M - 1) for j in range(M)]).reshape(
            (-1, 1)
        )  # resistance rates
        self.t = 0  # time
        self.mu = mu  # death rate
        self.gamma = gamma  # recovery rate
        self.ksi = self.build_ksi(
            N, C, M, self.omega
        )  # ksi[i,j,k,l] transmission rate between R[k,l] and I[i,j] .
        self.sigma = sigma  # noise standard deviation
        self.R0_max = R0_max  # max of R0
        self.h = h  # threshold for very small values
        self.T = [self.t]  # list of time
        P = np.zeros((self.N, self.M, self.N, self.M))  # gaussian kernel
        for i in range(N):
            for j in range(M):
                for k in range(N):
                    for l in range(M):
                        P[i, j, k, l] = (1 / (2 * np.pi * sigma**2)) * np.exp(
                            -((i - k) ** 2 + (j - l) ** 2) / (2 * sigma**2)
                        )
        self.P = P
        self.verbose = verbose

    def init_I(self, i0, j0, prop):
        I = np.zeros((self.N, self.M))
        I[i0, j0] = prop
        return I

    def init_R(self, i0, j0, prop):
        R = np.zeros((self.N, self.M))
        R[i0, j0] = prop
        return R

    def iter(self):

        # Get values at t-1
        S_prev = self.population.S[-1]
        I_prev = self.population.I[-1]
        V_prev = self.population.V[-1]
        R_prev = self.population.R[-1]
        D_prev = self.population.D[-1]
        i_prev = self.population.i[-1]
        r_prev = self.population.r[-1]

        # Update Susceptible (-n(t) could give negative values)
        self.population.S.append(
            max(
                S_prev
                - self.n(self.t)
                - self.eta(self.t)
                * S_prev
                * (np.sum(np.multiply(I_prev, self.beta @ np.ones((1, self.M))))),
                0,
            )
        )

        # Update Vaccinated (+n(t) could give >1 values)
        if S_prev <= self.h:
            self.population.V.append(V_prev)  # no sum to 1 otherwise
        else:
            self.population.V.append(
                min(
                    V_prev
                    + self.n(self.t)
                    - self.eta(self.t)
                    * (np.sum(np.multiply(I_prev, self.beta @ self.omega.T)))
                    * V_prev,
                    1,
                )
            )

        # Update Recovered
        R_next = (
            R_prev
            + self.gamma * I_prev
            - self.eta(self.t) * np.multiply(dotdot(permut(self.ksi), I_prev), R_prev)
        )

        R_next = np.clip(R_next, 0, 1)
        self.population.R.append(R_next)
        self.population.r.append(np.sum(self.population.R[-1]))

        # Update Dead
        self.population.D.append(D_prev + self.mu * i_prev)

        # Update Infected (and make sure this sums to 1) #TODO:why don't we vaccinate the recovered?
        # Compute the number of infected people by deduction to sum to 1
        infected_prop = 1 - (
            self.population.D[-1]
            + np.sum(self.population.R[-1])
            + self.population.S[-1]
            + self.population.V[-1]
        )
        if infected_prop <= 0:
            self.population.I.append(np.zeros(I_prev.shape))
        else:
            I_next = (
                I_prev
                + np.multiply(I_prev, self.beta @ np.ones((1, self.M))) * S_prev
                + self.eta(self.t)
                * (np.multiply(I_prev, self.beta @ self.omega.T))
                * V_prev
                - (self.mu + self.gamma) * I_prev
                - self.eta(self.t) * np.multiply(dotdot(self.ksi, R_prev), I_prev)
                + psi(dotdot(self.P, I_prev) - I_prev, self.h)
            )
            I_next = np.clip(I_next, 0, 1)
            if np.sum(I_next) == 0:
                self.population.I.append(np.zeros(I_next.shape))
            else:
                self.population.I.append(infected_prop / np.sum(I_next) * I_next)
        self.population.i.append(max(0, min(infected_prop, 1)))

        if self.verbose:

            print(f"Iteration {self.t} has just finished")
            print(
                f"The total proportion of people is {S_prev+i_prev+r_prev+D_prev+V_prev}"
            )
            print("--------------------------------------")

        self.t += 1
        self.T.append(self.t)

    def build_ksi(self, N, C, M, omega):
        # Initialize ksi

        ksi_2D = np.array(
            [[C / M * max(omega[j] - omega[i]) for j in range(M)] for i in range(M)]
        )
        ksi = np.tile(ksi_2D, (N, N)).reshape(N, ksi_2D.shape[0], N, ksi_2D.shape[1])
        for i in range(ksi.shape[0]):
            for j in range(ksi.shape[1]):
                for k in range(ksi.shape[2]):
                    for l in range(ksi.shape[3]):
                        assert (
                            ksi[i, j, k, l] == ksi[0, j, 0, l]
                        ), "The ksi tensor is not acceptable by crit 1"
                        assert (
                            ksi[i, j, k, l] >= ksi[i, j, k, max(l - 1, 0)]
                        ), "The ksi tensor is not acceptable by crit 2"
        return ksi

    def plot(self, t_max, animation=False):
        # Seed noise
        np.random.seed(1)
        if not animation:
            if self.verbose:
                print("The epidemic starts spreading")
            for t in range(t_max - 1):
                self.iter()
                if self.verbose:
                    plt.imshow(self.population.I[-1])
                    plt.colorbar()
                    plt.show()

            plt.plot(self.T, self.population.S, label="Susceptible", color="blue")
            plt.plot(self.T, self.population.D, label="Dead", color="black")
            plt.plot(self.T, self.population.V, label="Vaccinated", color="g")
            plt.plot(
                self.T,
                [np.sum(frame) for frame in self.population.I],
                label="Infected",
                color="red",
            )
            plt.plot(
                self.T,
                [np.sum(frame) for frame in self.population.R],
                label="Recovered",
                color="yellow",
            )
            plt.xlabel("Time")
            plt.ylabel("Proportion of population")
            plt.legend()
            plt.show()
        else:

            def animate(i):
                print(f"t={i}")
                self.iter()

                # erase previous plot
                ax.cla()

                # draw point's current position
                for key, value in self.population.__dict__.items():
                    if key in ["R", "I"]:
                        continue

                    ax.plot(self.T, value, label=labels[key], color=colors[key])

                # fix axes limits
                # ax.set_xlim(0, t_max)
                ax.set_ylim(-0.1, 1)
                ax.set_ylabel("Proportion of population")
                ax.set_xlabel("Time")
                ax.legend(loc="upper right")

            N_frames = t_max

            # generate figure and axis
            fig, ax = plt.subplots(figsize=(10, 7))
            colors = {
                "S": "blue",
                "D": "black",
                "i": "red",
                "r": "yellow",
                "V": "green",
            }
            labels = {
                "S": "Susceptible",
                "D": "Dead",
                "i": "Infected",
                "r": "Recovered",
                "V": "Vaccinated",
            }
            # define the animation
            ani = FuncAnimation(
                fig=fig, func=animate, interval=20, frames=N_frames, repeat=False
            )

            # Set the size of the figure
            fig.set_size_inches(8, 6)
            # show the animation
            manager = plt.get_current_fig_manager()

            # Set the position of the window
            manager.window.wm_geometry("+50+50")
            plt.show(block=True)
