import autograd.numpy as np
from autograd import elementwise_grad as egrad
from autograd import jacobian
from typing import Callable


class fit:
    """
    Fit class used for analyzing nonlinear regression with
    automatic differentiation.
    """

    def __init__(
        self,
        res_func: Callable,
        bestfit_pars: np.ndarray,
        extra_pars: dict,
        objfun: Callable = None,
        sig: float = None,
    ):
        """
        Initialize fit instance.

        Parameters
        ----------
        res_fun
            Function that calculates residuals. Accepts a vector of the shape
            of`bestfit_pars` and the dictionary `extra_pars`.
        bestfit_pars
            1 dimensional `numpy.ndarray` of best fit parameters.
        extra_pars
            Dictionary containing parameters required to compute the residuals
            but not estimated during the fitting. Can include data to fit,
            known parameters in the model etc.
        objfun
            Objective function of the model. If none provided, 0.5 * sum
            of squared errors is assumed.
        sig
            Standard deviation of the data. If none provided, computed using
            formula given below.

        Notes
        -----
        `sig`, if not provided, is estimated as sqrt(SSE)/(m-n) where SSE is
        the sum of squared errors, m = number of data points and n = number of
        parameters according to [1]_.

        References
        ----------
        .. [1] Ritz, C., & Streibig, J. C. (2009). Nonlinear Regression with R.
           https://doi.org/10.1007/978-0-387-09616-2
        """

        assert isinstance(bestfit_pars, np.ndarray)
        try:
            temp_res = res_func(bestfit_pars, extra_pars)
            self._m = len(temp_res)
        except:
            print("res_func was not callable with bestfit_pars and extra_pars")
            return
        self._bestfit_pars = bestfit_pars
        self._res_func = res_func
        self._extra_pars = extra_pars
        if objfun is None:
            self._objfun = sse
        else:
            self._objfun = objfun
        if sig is None:
            self._sig = self.get_sig()

    def residual(self, x=None) -> np.ndarray:
        """Get vector of residuals.

        Call the provided residual function with parameters in `x`.

        Parameters
        ---------
        x
            The `np.ndarray` of parameters used to compute the residual.

        Returns
        -------
        res
            The `np.ndarray` of residuals.
        """
        if x is None:
            x = self._bestfit_pars
        res = self._res_func(x, self._extra_pars)
        return res

    def get_J(self, x=None) -> np.ndarray:
        """Jacobian of residuals.

        Residuals (prediction - data) of the fit of interest.

        Parameters
        ----------
        x
            The `np.ndarray` of paramaters used to compute the Jacobian.

        Returns
        -------
        J
            Jacobian of residuals.
        """
        if x is None:
            x = self._bestfit_pars
        J = jacobian(lambda pars: self._res_func(pars, self._extra_pars))(x)
        return J

    def get_H(self, x=None) -> np.ndarray:
        """Hessian of objective function.

        Hessian of the objective function is the Jacboian of the gradient
        of the objective funciton.

        Parameters
        ----------
        x
            The `np.ndarray` of paramaters used to compute the Hessian.

        Returns
        -------
        H
            Hessian of the objective function.
        """
        if x is None:
            x = self._bestfit_pars
        H = jacobian(
            egrad(lambda pars: self._objfun(
                self._res_func, pars, self._extra_pars))
        )(x)
        return H

    def get_sig(self) -> float:
        """Standard deviation of the fit.

        Estimate standard deviation from the residual vector.

        Returns
        -------
        sig
            Estimated standard deviation.
        """
        x = self._bestfit_pars
        m = self._m
        n = len(self._bestfit_pars)
        res = self.residual(x)
        sig = np.sqrt(np.matmul(res.transpose(), res) / (m - n))
        return sig

    def get_vcov(self, x=None) -> np.ndarray:
        """Variance-covariance matrix of parameters.

        Estimate variance-covariance matrix of the provided parameters.

        Parameters
        ----------
        x
            The `np.ndarray` of parameters used to compute the Hessian.

        Returns
        -------
        vcov
            Variance-covariance matrix of the provided parameters.
        """
        if x is None:
            x = self._bestfit_pars
        sig = self._sig
        H = self.get_H(x)
        Hinv = np.linalg.inv(H)
        vcov = (sig ** 2) * Hinv
        return vcov

    def get_sd_bf(self):
        """Standard deviation of best-fit parameters.

        Get the standard deviation of the best-fit parmeters.

        Returns
        -------
        sd_bf
            Standard deviations of the best-fit parameters.
        """
        vcov = self.get_vcov(x=self._bestfit_pars)
        sd_bf = np.sqrt(np.diag(vcov))
        return sd_bf


def sse(res_func: Callable, par_list: np.ndarray, extra_pars: dict) -> float:
    """Sum of squared residuals.

    From the residual function, parameters and the auxiliary parameters,
    compute the sum of squared error residuals.

    Parameters
    ----------
    res_func
        Function that calculates the residuals.
    par_list
        1 dimensional `numpy.ndarray` of fitted parameters.
    extra_pars
        Dictionary containing parameters required to compute the residuals
        but not estimated during the fitting. Can include data to fit,
        known parameters in the model etc.
    """
    residuals = res_func(par_list, extra_pars)
    sse = 0.5 * np.matmul(residuals.transpose(), residuals)
    return sse
