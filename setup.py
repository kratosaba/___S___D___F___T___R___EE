import numpy
import setuptools  # important
from Cython.Build import cythonize
from setuptools import setup, Extension, Command


# set up the library with name Tringle
setup(ext_modules = cythonize(Extension(
           name = "Tringle",  
           sources=["triangle_hash.pyx"],     
           language="c++",
           include_dirs=[numpy.get_include()]                      
      )
      ))


