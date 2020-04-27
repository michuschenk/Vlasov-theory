from cython_gsl cimport *
import cython

import numpy as np
cimport numpy as np

from libc.math cimport cos, sin, M_PI
from libc.stdlib cimport malloc, free

from cython.parallel import prange
from openmp cimport omp_get_max_threads, omp_get_thread_num

ctypedef double * double_ptr
ctypedef void * void_ptr


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double integrand(double x, void * params) nogil:
    """ Q0: main tune, Qs: synchrotron tune, rb: airbag beam 'radius',
    R: machine radius, eta: slippage factor, Qp: first-order chromaticity,
    Qpp: second-order chromaticity, p: summand index, l: azimuthal mode number """
    # TODO: write which integrand we are dealing with here.
    cdef double f
    cdef double Q0, Qs, rb, R, eta, Qp, Qpp, p, l

    Q0 = (<double_ptr> params)[0]
    Qs = (<double_ptr> params)[1]
    rb = (<double_ptr> params)[2]
    R = (<double_ptr> params)[3]
    eta = (<double_ptr> params)[4]
    Qp = (<double_ptr> params)[5]
    Qpp = (<double_ptr> params)[6]
    p = (<double_ptr> params)[7]
    l = (<double_ptr> params)[8]

    f = (-l*x + (Q0+p+l*Qs)*rb/R*cos(x) + Qp/(eta*R)*rb*(1.-cos(x)) -
         Qpp*Qs*rb*rb/(4.*eta*eta*R*R)*sin(x)*cos(x))

    return f


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double hl2_re(double x, void * params) nogil:
    # Take real part and normalise
    cdef double f
    f = cos(integrand(x, params)) / (2.*M_PI)
    return f


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double hl2_im(double x, void * params) nogil:
    # Take imag part and normalise
    cdef double f
    f = sin(integrand(x, params)) / (2.*M_PI)
    return f