from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("cpp_get_evidence.pyx")
)

