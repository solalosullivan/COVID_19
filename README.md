Study of COVID-19 like virus epidemic thanks to "Modeling the emergence of vaccine-resistant variants with
Gaussian convolution" paper

DOI : https://doi.org/10.1101/2021.07.07.21259916

Run and set all parameters in main.py
Epidemic model in model.py

In order to see the epidemic spreading, set animation=True in main.py. 
To see the mutations spreading, go to plots.ipynb
Models are saved in /models with the chosen eta. We haven't tried all three initializations yet.

The optimization is done in main_opti.py you can select the desired parameters at the bottom.
M and N have been set so that it doesn't take ages to run, feel free to modify them.
In case where the opti result is not satisfyng :
    - try an higher C for annealing
    - increase nb_it for genetic

For now our opti always gives max parameters at any time, that is why no result has been saved so far. 
We will try to obtain different results and comment them. 



