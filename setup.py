from setuptools import Extension, setup
import numpy as np
from Cython.Build import cythonize


ext_modules = [
    Extension("integrators",
              sources=["integrators.pyx"],
              libraries=["m"],
              include_dirs=[np.get_include()],
              extra_compile_args=['-O3', '-Xpreprocessor', '-fopenmp'],
              extra_link_args = ['-Xpreprocessor', '-fopenmp']
              )
]

setup(name="integrators",
      ext_modules=cythonize(ext_modules),
      include_dirs=[np.get_include()])
