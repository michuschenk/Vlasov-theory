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
