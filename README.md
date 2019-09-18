# adnls
Using automatic differentiation as implemented in the `autograd` package for analyzing non-linear curve fitting. Presently finds the Jacobian, Hessian, variance covariance matrix of fitted parameters for user-defined non-linear model.

## Quick start/Example

Example taken from [Niclas Börlin's lecture slides](https://www8.cs.umu.se/kurser/5DA001/HT07/lectures/lsq-handouts.pdf), which also has a clear explanation of the concepts involved. First the model and residual function are defined.

```python
from adnls import fit
import autograd.numpy as np

# The model used for nonlinear regression
def model(t, p):
    y = p[0] * np.exp(p[1] * t)
    return y

# The residual function used by the `fit` class
def res(p, extra_pars):
    return model(extra_pars["t"], p) - extra_pars["data"]
```
Then define the data to be stored in the `extra_pars` variable, which is a dictionary. Best fitting values are assumed to have been found by some other method.

```python
extra_pars = {}
extra_pars["t"] = np.array([1, 2, 4])
extra_pars["data"] = np.array([3, 4, 6])
x_best = np.array((2.4893671, 0.2210597), dtype=np.float64)
```

Define the instance of the `fit` class and then call the requisite functions.

```python
this_fit = fit(
    res_func=res,
    bestfit_pars=x_best,
    extra_pars=extra_pars,
)  # create instance of fit class
print("Hessian")
print(this_fit.get_H(x_best))
print(this_fit.get_H())
print("Jacobian")
print(this_fit.get_J(x_best))
print("Residual")
print(this_fit.residual())
print(this_fit.residual(x_best))
print("Vcov matrix")
print(this_fit.get_vcov())
print("Sd of best-fit paramters")
print(this_fit.get_sd_bf())
```
### Output

```
Hessian
[[  9.83906461  74.29758072] 
 [ 74.29758072 651.85399352]]
[[  9.83906461  74.29758072] 
 [ 74.29758072 651.85399352]]
Jacobian
[[ 1.2473979   3.10523129]
 [ 1.55600152  7.74691796]
 [ 2.42114072 24.10843219]]
Residual
[ 0.10523129 -0.12654102  0.02710805]
[ 0.10523129 -0.12654102  0.02710805]
Vcov matrix
[[ 0.02029685 -0.00231341]
 [-0.00231341  0.00030636]]
Sd of best-fit paramters
[0.142467   0.01750314]
```

## Features
1. Compute Jacobian of the residuals with respect to fitted parameters using automatic differentiation.
2. Compute Hessian of the objective function with respect to fitted parameters using automatic differentiation.
3. Estimate standard deviation if it is not known.
4. Estimate variance covariance matrix of the fitted parmeters, derived the Hessian found by automatic differentiation.
5. Estimate the standard deviations of the fitted parameters, derived the Hessian found by automatic differentiation.

## Requirements
Install the `autograd` package with `pip install autograd`. See [here](https://github.com/HIPS/autograd) for further instructions.

## Caveats
1. Model must use `autograd.numpy` and `autograd.scipy` where applicable. 
2. Only the subset of `numpy` and `scipy` implemented by the `autograd` package will work. See [here](https://github.com/HIPS/autograd/blob/master/docs/tutorial.md#supported-and-unsupported-parts-of-numpyscipy) for list of things that will work.

# To do
1. Investigate differential equation models.
2. Add a summary like the `nls` function in R, with significance of parameters reported.