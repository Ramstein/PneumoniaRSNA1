# --------------------------------------------------------
# Deformable Convolutional Networks
# Copyright (c) 2017 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Modified by Yuwen Xiong
# --------------------------------------------------------
# Based on:
# py-faster-rcnn
# Copyright (c) 2016 by Contributors
# Licence under The MIT License
# py-faster-rcnn (https://github.com/rbgirshick/py-faster-rcnn)
# --------------------------------------------------------

from distutils.extension import Extension

import numpy as np
from Cython.Distutils import build_ext
# from distutils.core import setup
from setuptools import setup

# change for windows, by MrX
nvcc_bin = 'nvcc.exe'
lib_dir = 'lib/x64'

import distutils.msvc9compiler

distutils.msvc9compiler.VERSION = 14.0

# Obtain the numpy include directory.  This logic works across numpy versions.
try:
    numpy_include = np.get_include()
except AttributeError:
    numpy_include = np.get_numpy_include()

ext_modules = [
    # unix _compile: obj, src, ext, cc_args, extra_postargs, pp_opts
    Extension(
        "bbox",
        sources=["bbox.pyx"],
        extra_compile_args={},
        include_dirs=[numpy_include]
    ),
]

setup(
    name='fast_rcnn',
    ext_modules=ext_modules,
    # inject our custom trigger
    cmdclass={'build_ext': build_ext},
)
