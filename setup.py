#!/usr/bin/env python

from setuptools import setup, find_packages

install_reqs = []
with open('requirements.txt') as f:
    install_reqs.append(f.read().splitlines())


def setup_package():
    setup(
        name="rBCM",
        description="A sample robust Bayesian Committee Machine \
                     implementation",
        author="Lucas J. Kolstad",
        author_email="lkolstad@uw.edu",
        version="0.2.0",
        license="See LICENSE file.",
        packages=find_packages(),
        install_requires=install_reqs
    )


if __name__ == '__main__':
        setup_package()
