#!/usr/bin/env python

from setuptools import setup, find_packages


def get_install_reqs():
    install_reqs = []
    with open('requirements.txt') as f:
        install_reqs.append(f.read().splitlines())
    return install_reqs


def get_long_description():
    with open('README.rst') as f:
        desc = f.read()
    return desc


def setup_package():
    setup(
        name="rBCM",
        description="A robust Bayesian Committee Machine Regressor.",
        author="Lucas J. Kolstad",
        author_email="lkolstad@uw.edu",
        version="0.2.0",
        license="MIT",
        packages=find_packages(),
        install_requires=get_install_reqs(),
        keywords="statistics gaussian process bayesian regression committee",
        long_description=get_long_description(),
        classifiers=[
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: MIT License",
            "Natural Language :: English",
            "Operating System :: Unix",
            "Programming Language :: Python",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "Topic :: Scientific/Engineering :: Information Analysis",
            "Topic :: Scientific/Engineering :: Mathematics",
        ],
    )


if __name__ == '__main__':
        setup_package()
