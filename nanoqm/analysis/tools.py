"""Miscellaneous functions to analyze the simulation results.

Index
-----
.. currentmodule:: nanoqm.analysis.tools
.. autosummary::

API
---

"""

import os
from typing import List, Tuple

import numpy as np
import pyparsing as pa
from scipy.optimize import curve_fit

from ..common import fs_to_cm, h2ev, hbar, r2meV

""" Functions to fit data """


def gauss_function(x: float, sigma: float) -> np.ndarray:
    """Compute a Gaussian function used for fitting data."""
    return np.exp(-0.5 * (-x / sigma) ** 2)


def lorentzian_function(x_L, sigma_L, amplitude_L):
    """Compute a Lorentzian function used for fitting data."""
    return amplitude_L * sigma_L**2 / (sigma_L**2 + x_L ** 2)


def exp_function(x, x0, amplitude):
    """Compute an Exponential function used for fitting data."""
    return amplitude * np.exp(-x / x0)


def sine_function(t_phonon, amplitude, offset, phase, n_periods, dt):
    """Compute a sinusoidal function used for fitting data."""
    t = np.arange(0, n_periods * t_phonon, dt)
    # (gap) energy
    y = offset + amplitude * (np.sin(2 * np.pi * t / t_phonon + phase))
    y_mean = np.mean(y)
    y_dummy = y / h2ev    # to delete ...
    return y_dummy, y, y_mean, t


def sqrt_func(x, a):
    """Compute a square root function used for fitting data."""
    return a * np.sqrt(x)


def func_conv(x_real: np.ndarray, x_grid: np.ndarray, delta: float) -> np.ndarray:
    """Compute a convolution on a grid using a Gaussian function."""
    return np.exp(-2 * (x_grid - x_real) ** 2 / delta ** 2)


def convolute(x: np.ndarray, y: np.ndarray, x_points: np.ndarray, sigma: float) -> np.ndarray:
    """Convolute a spectrum on a grid of x_points.

    You need as input x, y and the grid where to convolute.
    """
    # Compute gaussian prefactor
    prefactor = np.sqrt(2.0) / (sigma * np.sqrt(np.pi))
    # Convolute spectrum over grid
    y_points = prefactor * np.stack([
        np.sum(y * func_conv(x, x_point, sigma)) for x_point in x_points
    ])
    return y_points


""" Useful functions to compute autocorrelation, dephasing, etc. """


def autocorrelate(f: np.ndarray) -> Tuple[float, float]:
    """Compute the un-normalized and normalized autocorrelation of a function."""
    d_f = f - f.mean()
    d_f2 = np.append(d_f, d_f, axis=0)
    # Compute the autocorrelation function
    uacf = np.correlate(d_f, d_f2, "valid")[:d_f.size] / d_f.size
    # Compute the normalized autocorrelation function
    nacf = uacf / uacf[0]
    return uacf, nacf


def spectral_density(f, dt):
    """Fourier Transform of a given function f using a dense grid with 100000 points.

    In the case of a FFT of a normalized autocorrelation function,
    this corresponds to a spectral density
    """
    n_pts = 100000
    f_fft = abs(1 / np.sqrt(2 * np.pi) * np.fft.rfft(f, n_pts) * dt) ** 2
    # Fourier Transform of the time axis
    freq = np.fft.rfftfreq(n_pts, dt)
    # Conversion of the x axis (given in cycles/fs) to cm-1
    freq = freq * fs_to_cm
    return f_fft, freq


def dephasing(f: np.ndarray, dt: float):
    """Compute the dephasing time of a given function.

        f            = energies, in eV  (if other function/unit: then the line_broadening will not be in eV)
        dt           = time step, in fs

    Use the optical response formalisms:
    S. Mukamel, Principles of Nonlinear Optical Spectroscopy, 1995
    About the implementation we use the 2nd order cumulant expansion.
    See also eq. (2) in : Kilina et al. Phys. Rev. Lett., 110, 180404, (2013)
    To calculate the dephasing time tau we fit the dephasing function to a
    gaussian of the type : exp(-0.5 * (-x / tau) ** 2)
    """
    # Conversion of hbar to hartree * fs
    hbar_au = hbar / h2ev
    ts = np.arange(f.shape[0]) * dt
    cumu_ii = np.asarray([np.trapz(f[0:i + 1], dx=(dt / hbar), axis=0) for i in range(ts.size)])
    cumu_i = np.asarray([np.trapz(cumu_ii[0:i + 1], dx=(dt / hbar), axis=0)
                         for i in range(ts.size)])
    deph = np.exp(-cumu_i)

    return deph, ts


def fit_dephasing(fit_func, deph, ts, res, t_deph_guess):
    """Work in progress (?).

    fit_func = 0 for Gaussian fit, or 1 for exponential fit
    deph     = dephasing function vs. time "
    ts       = time, in fs
    res      = factor by which to increase the time resolution of ts for the fit
    t_deph_guess = initial guess, in fs
    """
    np.seterr(over='ignore')

    if fit_func == 0:
        # fit with Gaussian
        popt, pcov = curve_fit(gauss_function, ts, deph, p0=(t_deph_guess, deph[0]))    # [0]
        ts_fit = np.arange(res * deph.shape[0]) * dt / res
        deph_fit = popt[1] * np.exp(-0.5 * (-ts_fit / popt[0]) ** 2)
        deph_time = popt[0]                                           # in fs (defined as standard deviation of a Gaussian)
        e_fwhm = std_to_fwhm * hbar / deph_time                       # FWHM in eV
        perr = np.sqrt(np.diag(pcov))
        deph_time_err = perr[0]                                            # error (standard deviation) for the deph. time
        e_fwhm_err = deph_time_err * std_to_fwhm * hbar / (deph_time ** 2)  # error (standard deviation) for the FWHM
    elif fit_func == 1:
        # fit with exponential
        popt, pcov = curve_fit(exp_function, ts, deph, p0=(t_deph_guess, deph[0]))  # [0]
        ts_fit = np.arange(res * deph.shape[0]) * dt / res
        deph_fit = popt[1] * np.exp(-ts_fit / popt[0])
        deph_time = popt[0]                                           # in fs (defined as the exp. time constant)
        e_fwhm = 2 * hbar / deph_time                                 # FWHM in eV
        perr = np.sqrt(np.diag(pcov))
        deph_time_err = perr[0]                                       # error (standard deviation) for the deph. time
        e_fwhm_err = deph_time_err * 2 * hbar / (deph_time ** 2)      # error (standard deviation) for the FWHM

    return ts_fit, deph_fit, deph_time, deph_time_err, e_fwhm, e_fwhm_err


def read_couplings(path_hams, ts):
    """Read the non adiabatic coupling vectors from the files generated for the NAMD simulations."""
    files_im = [os.path.join(path_hams, f'Ham_{i}_im')
                for i in range(ts)]
    xs = np.stack(np.loadtxt(fn) for fn in files_im)
    return xs * r2meV  # return energies in meV


def read_energies(path_hams, ts):
    """Read the molecular orbital energies of each state.

    The target files are generated for the NAMD simulations.
    """
    files_re = [os.path.join(path_hams, f'Ham_{i}_re')
                for i in range(ts)]
    xs = np.stack(np.diag(np.loadtxt(fn)) for fn in files_re)
    return xs * r2meV / 1000  # return energies in eV


def read_energies_pyxaid(path, fn, nstates, nconds):
    """Read the molecular orbital energies of each state from the output files generated by PYXAID."""
    inpfile = os.path.join(path, fn)
    cols = tuple(range(5, nstates * 2 + 5, 2))
    xs = np.stack(np.loadtxt(f'{inpfile}{j}', usecols=cols)
                  for j in range(nconds)).transpose()
    # Rows = timeframes ; Columns = states ; tensor = initial conditions
    xs = xs.swapaxes(0, 1)
    return xs


def read_pops_pyxaid(path, fn, nstates, nconds):
    """Read the population of each state from the output files generated by PYXAID."""
    inpfile = os.path.join(path, fn)
    cols = tuple(range(3, nstates * 2 + 3, 2))
    xs = np.stack(np.loadtxt(f'{inpfile}{j}', usecols=cols)
                  for j in range(nconds)).transpose()
    # Rows = timeframes ; Columns = states ; tensor = initial conditions
    xs = xs.swapaxes(0, 1)
    return xs


def parse_list_of_lists(xs: str) -> List[List[int]]:
    """Parse a list of list of integers using pyparsing."""
    enclosed = pa.Forward()  # Parser to be defined later
    natural = pa.Word(pa.nums)  # Natural Number
    # Nested Grammar
    nestedBrackets = pa.nestedExpr(pa.Suppress(
        '['), pa.Suppress(']'), content=enclosed)
    enclosed << (natural | pa.Suppress(',') | nestedBrackets)
    try:
        rs = enclosed.parseString(xs).asList()[0]
        return list(map(lambda x: list(map(int, x)), rs))
    except pa.ParseException:
        raise RuntimeError("Invalid Macro states Specification")
