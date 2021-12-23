"""This file contains a Numba-accelerated implementation of the digamma
function, taken from the source of the spycial Python package.

This was due to issues with re-importing causing the following crash.
LLVM ERROR: Symbol not found: .numba.unresolved$_ZN7spycial3erf14_erf_erfc$2441Ed28Literal$5bbool$5d$28False$29

Spycial is licenced with the BSD-3 license.

BSD 3-Clause License

Copyright (c) 2018, Josh Wilson
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Implementation of the digamma function. The original code is from
Cephes:
Cephes Math Library Release 2.8:  June, 2000
Copyright 1984, 1987, 1992, 2000 by Stephen L. Moshier
Code for the rational approximation on [1, 2] is from Boost, which is:
(C) Copyright John Maddock 2006.
Use, modification and distribution are subject to the
Boost Software License, Version 1.0. (See accompanying file
LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
"""
import numpy as np
from numba import njit, vectorize
from numba.extending import intrinsic
from numba.types import Array, float64
from llvmlite import ir


# Harmonic numbers minus the Euler-Mascheroni constant
HARMONIC = np.array(
    [
        -0.5772156649015328606065121,
        0.4227843350984671393934879,
        0.9227843350984671393934879,
        1.256117668431800472726821,
        1.506117668431800472726821,
        1.706117668431800472726821,
        1.872784335098467139393488,
        2.015641477955609996536345,
        2.140641477955609996536345,
        2.251752589066721107647456,
    ]
)

ASYMP = np.array(
    [
        8.33333333333333333333e-2,
        -2.10927960927960927961e-2,
        7.57575757575757575758e-3,
        -4.16666666666666666667e-3,
        3.96825396825396825397e-3,
        -8.33333333333333333333e-3,
        8.33333333333333333333e-2,
    ]
)

RAT_NUM = np.array(
    [
        -0.0020713321167745952,
        -0.045251321448739056,
        -0.28919126444774784,
        -0.65031853770896507,
        -0.32555031186804491,
        0.25479851061131551,
    ]
)

RAT_DENOM = np.array(
    [
        -0.55789841321675513e-6,
        0.0021284987017821144,
        0.054151797245674225,
        0.43593529692665969,
        1.4606242909763515,
        2.0767117023730469,
        1.0,
    ]
)


@intrinsic
def _fma(typing_context, x, y, z):
    """Compute x * y + z using a fused multiply add."""
    sig = float64(float64, float64, float64)

    def codegen(context, builder, signature, args):
        ty = args[0].type
        mod = builder.module
        fnty = ir.types.FunctionType(ty, [ty, ty, ty])
        fn = mod.declare_intrinsic("llvm.fma", [ty], fnty)
        ret = builder.call(fn, args)
        return ret

    return sig, codegen


@njit(float64(Array(float64, 1, "C", readonly=True), float64))
def _devalpoly(coeffs, x):
    """Evaluate a polynomial using Horner's method."""
    res = coeffs[0]

    for j in range(1, len(coeffs)):
        res = _fma(res, x, coeffs[j])

    return res


@njit("float64(float64)")
def _digamma_rational(x):
    """Rational approximation on [1, 2] taken from Boost.
    For the approximation, we use the form
    digamma(x) = (x - root) * (Y + R(x-1))
    where root is the location of the positive root of digamma, Y is a
    constant, and R is optimised for low absolute error compared to Y.
    Maximum deviation found: 1.466e-18
    At double precision, max error found: 2.452e-17
    """
    Y = np.float32(0.99558162689208984)
    root1 = 1569415565.0 / 1073741824.0
    root2 = (381566830.0 / 1073741824.0) / 1073741824.0
    root3 = 0.9016312093258695918615325266959189453125e-19

    g = x - root1
    g -= root2
    g -= root3
    r = _devalpoly(RAT_NUM, x - 1.0) / _devalpoly(RAT_DENOM, x - 1.0)

    return g * Y + g * r


@njit("float64(float64)")
def _digamma(x):
    res = 0.0

    if np.isnan(x) or x == np.inf:
        return x
    elif x == -np.inf:
        return np.nan
    elif x == 0:
        return np.copysign(np.inf, -x)
    elif x < 0.0:
        # Argument reduction before evaluating tan(πx).
        r = np.fmod(x, 1.0)
        if r == 0.0:
            return np.nan
        pir = np.pi * r
        # Reflection formula
        res = -np.pi * np.cos(pir) / np.sin(pir) - 1.0 / x
        x = -x

    if x <= 10.0:
        if x == np.floor(x):
            # Exact values for for positive integers up to 10
            res += HARMONIC[np.intc(x) - 1]
            return res
        # Use the recurrence relation to move x into [1, 2]
        if x < 1.0:
            res -= 1.0 / x
            x += 1.0
        elif x < 10.0:
            while x > 2.0:
                x -= 1.0
                res += 1.0 / x

        res += _digamma_rational(x)
        return res

    # We know x is large, use the asymptotic series.
    if x < 1.0e17:
        z = 1.0 / (x * x)
        y = z * _devalpoly(ASYMP, z)
    else:
        y = 0.0
    res += np.log(x) - (0.5 / x) - y
    return res


@vectorize(["float64(float64)"], nopython=True)
def digamma(x):
    """Digamma function.
    Parameters
    ----------
    x : array-like
        Points on the real line
    out : ndarray, optional
        Output array for the values of `digamma` at `x`
    Returns
    -------
    ndarray
        Values of `digamma` at `x`
    """
    return _digamma(x)
