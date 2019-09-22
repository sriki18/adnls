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
3. Differential equation models don't appear to work at the moment. For example, consider the differential equation variant of the model above:
   ```python
   from adnls import fit
   import autograd.numpy as np
   from autograd.scipy.integrate import odeint
   # from scipy.integrate import odeint


   def derivative(y, t, p):
       dy = p[0] * p[1] * np.exp(p[1] * t)
       return dy


   def model(t, p):
       y0 = np.array([0])
       sol = odeint(derivative, y0, t, args=(p,))
       print(sol)
       return sol[1:, 0] + p[0]


   def res(p, extra_pars):
       return model(extra_pars["t"], p) - extra_pars["data"]
       # [ 0.10523129 -0.12654102  0.02710805]


   if __name__ == "__main__":
       extra_pars = {}
       extra_pars["t"] = np.array([0, 1, 2, 4])
       extra_pars["data"] = np.array([3, 4, 6])
       x_best = np.array((2.4893671, 0.2210597), dtype=np.float64)
       print(res(x_best, extra_pars))
       this_fit = fit(
           res_func=res,
           bestfit_pars=x_best,
           extra_pars=extra_pars,
       )
       print("Jacobian")
       print(this_fit.get_J(x_best))
   ```
    This fails with     
   ```
    ...
     File "C:\Users\username\Documents\projects\adnls\adnls.py", line 109, in <lambda>
       J = jacobian(lambda pars: self._res_func(pars, self._extra_pars))(x)
     File "diffeq.py", line 21, in res
       return model(extra_pars["t"], p) - extra_pars["data"]
     File "diffeq.py", line 15, in model
       sol = odeint(derivative, y0, t, args=(p,))
     File "C:\Users\username\Miniconda3\envs\ad\lib\site-packages\autograd\tracer.py", line 48, in f_wrapped
       return f_raw(*args, **kwargs)
     File "C:\Users\username\Miniconda3\envs\ad\lib\site-packages\scipy\integrate\odepack.py", line 244, in odeint
       int(bool(tfirst)))
   ValueError: setting an array element with a sequence.
    ```

    The model itself is fine, though. This can be verified by uncommeting `# from scipy.integrate import odeint` and commenting out `from autograd.scipy.integrate import odeint`. Not sure if it's a bug in my implementation.

### Work-around
If you must use a differential equation model, considering using the up-to-date [JAX library](https://github.com/google/jax). Presently there is an experimental ODE solver in JAX at https://github.com/google/jax/blob/master/jax/experimental/ode.py .
- But this only works on Linux/macOS. See [here](https://github.com/google/jax/issues/438) for discussion on porting JAX for Windows - apparently this can be done with Windows Subsystem for Linux (WSL). (The challenge was getting `jaxlib` to compile on Windows)


# To do
1. Try out WSL for differential equation models.