#!/usr/bin/env python

from setuptools import setup, find_packages
from Cython.Build import cythonize
import sys
import subprocess
import numpy as np

cython_sources = ["rBCM/weighting.pyx"]


def setup_package():
    setup(
        name="rBCM",
        description="A robust Bayesian Committee Machine implementation",
        author="Lucas J. Kolstad",
        author_email="lucaskolstad@gmail.com",
        version="0.0.1",
        license="See LICENSE file.",
        packages=find_packages(),
        ext_modules=cythonize(cython_sources),
        include_dirs=[np.get_include()]
    )


def cleanup_build():
    print("Cleaning up the project directory...")
    # Make sure you have version control first....
    subprocess.Popen('rm -rf build', shell=True, executable="bash")
    subprocess.Popen('find . -name "*.c" -type f -delete', shell=True, executable="bash")
    subprocess.Popen('find . -name "*.pyc" -type f -delete', shell=True, executable="bash")
    subprocess.Popen('find . -name "*.so" -type f -delete', shell=True, executable="bash")
    subprocess.Popen('rm -rf dist', shell=True, executable="bash")
    subprocess.Popen('rm -rf rBCM.egg-info', shell=True, executable="bash")
    subprocess.Popen('find . -name "__pycache__" -type d -delete', shell=True, executable="bash")
    subprocess.Popen('find . -name "tests/visuals/*" -type d -delete', shell=True, executable="bash")

    # do the normal cleaning too
    sys.argv[1] = "clean"


if __name__ == '__main__':
    arg = sys.argv[1].strip().lower()
    if arg == "clean" or arg == "c" or arg == "-c" or arg == "--clean":
        cleanup_build()
    else:
        args = sys.argv[1:]

        # Add convenience "-i" flag to build in place
        if args.count("-i") > 0:
            if args.count("build_ext") > 0 and args.count("--inplace") == 0:
                sys.argv.insert(sys.argv.index("build_ext") + 1, "--inplace")

            if args.count("install") > 0 and args.count('build_ext') == 0:
                sys.argv.append("build_ext")
                sys.argv.append("--inplace")
            sys.argv.remove("-i")

        if sys.platform == "win32" or sys.platform == "cygwin":
            sys.argv.insert("-c", "mingw32")

        setup_package()
