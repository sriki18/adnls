# adnls
Using automatic differentiation as implemented in the `autograd` package for analyzing non-linear curve fitting. Presently finds the Jacobian, Hessian, variance covariance matrix of fitted parameters for user-defined non-linear model.

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
1. Add a simple example.
2. Investigate differential equation models.
3. Add a summary like the `nls` function in R, with significance of parameters reported.