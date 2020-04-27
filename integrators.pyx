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

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def Hlp2(double Q0, double Qs, double rb, double R, double eta,
         double Qp, double Qpp, double l, int p_max):
    """ Serial implementation of (H_l^p)**2 function with parameters as
    explained above """

    cdef Py_ssize_t pp
    cdef np.ndarray[np.float64_t, ndim=1] results = (
        np.empty(2*p_max+1, dtype=np.float64))

    cdef double params[9]
    params[0] = Q0
    params[1] = Qs
    params[2] = rb
    params[3] = R
    params[4] = eta
    params[5] = Qp
    params[6] = Qpp
    params[8] = l

    cdef double ore, oim, erre, erim
    cdef gsl_function Fre
    cdef gsl_function Fim
    cdef gsl_integration_workspace * W
    W = gsl_integration_workspace_alloc(2000)

    for pp in range(-p_max, p_max+1, 1):
        params[7] = pp
        Fre.function = &hl2_re
        Fre.params = params
        Fim.function = &hl2_im
        Fim.params = params

        gsl_integration_qags(&Fre, 0, 2*M_PI, 0, 1e-7, 1000, W, &ore, &erre)
        gsl_integration_qags(&Fim, 0, 2*M_PI, 0, 1e-7, 1000, W, &oim, &erim)

    gsl_integration_workspace_free(W)

    return results
