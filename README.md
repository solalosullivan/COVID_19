Study of COVID-19 like virus epidemic thanks to "Modeling the emergence of vaccine-resistant variants with
Gaussian convolution" paper

DOI : https://doi.org/10.1101/2021.07.07.21259916

This code aims at making a simulation of a virus epidemic based on the paper model. The main paper breakthrough deals with
the virus mutation, which is the core of their model.

I. Create your environment and import requirements.txt

II. Set your chosen parameters in main.py 

Interesting parameters to modify :
- R0_max : the max transmission rate
- inf_prop : the initial part of population that is infected
- t_half_pop_vaccinated : the estimated time to vaccine half of the population (estimated because the vaccination rate is bounded)
- eta(t) : the restriction level applied to population

Run main.py

III. See the epidemic spreading curves animation

IV. Go to models/ 

See how the virus mutated and how the distribution of infected moved towards the strongest variants.





