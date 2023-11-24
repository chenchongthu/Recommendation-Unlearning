import os
from distutils.core import Extension, setup

import numpy as np
from Cython.Build import cythonize

pyx_directories = ["evaluator/cpp/"]

extensions = [Extension("*", ["*.pyx"], extra_compile_args=["-std=c++11"])]

pwd = os.getcwd()
for dir in pyx_directories:
    target_dir = os.path.join(pwd, dir)
    os.chdir(target_dir)
    setup(
        ext_modules=cythonize(extensions, language="c++"),
        include_dirs=[np.get_include()],
    )
    os.chdir(pwd)
