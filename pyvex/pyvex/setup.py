from setuptools import setup, Extension
import os

LIBPYVEX = 'libpyvex'

source = [os.path.join(LIBPYVEX, f) for f in os.listdir(LIBPYVEX) if f.endswith('.cpp') or f.endswith('.c')]

print(source)

setup(
    name="pyvex",
    description="Python bindings to CSIRO's vex file parser",
    author="Samuel Foster",
    url="https://github.com/ICRAR/pyvex",
    packages=['pyvex'],
    ext_modules=[
        Extension(LIBPYVEX, source, include_dirs=[LIBPYVEX], language='c++')
    ]
)
