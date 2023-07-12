# Welcome to `pynewmarkdisp` repo and docs

[![made-with-python](https://img.shields.io/badge/Made%20with-Python3-brightgreen.svg)](https://www.python.org/)   [![MIT License](https://img.shields.io/github/license/eamontoyaa/pynewmarkdisp.svg)](https://opensource.org/license/mit/)   [![PyPI repo](https://img.shields.io/pypi/v/pynewmarkdisp.svg)](https://pypi.org/project/pynewmarkdisp)

`pyNewmarkDisp` is an application software in Python to calculate permanent displacements in shallow earthquake-triggered landslides by the Newmark sliding block method.

In this repo and in the package [doumentation](https://eamontoyaa.github.io/pynewmarkdisp/) you will find the source code, instructions for installation, docstrings, examples of use, development history track, and references.

## Installation

It is suggested to create a virtual environment to install and use the program.

### Stable release

We recommend installing `pyNewmarkDisp` from [PyPI](https://pypi.org/project/pynewmarkdisp), as it will always install the most recent stable release.  To do so, run this command in your terminal:

    pip install pynewmarkdisp

### From sources

The sources for `pyNewmarkDisp` can be downloaded from the [Github repo](https://github.com/eamontoyaa/pyNewmarkDisp). You can clone the public repository running the following command:

    git clone git://github.com/eamontoyaa/pynewmarkdisp

Once you have a copy of the source code, you can install it with the following instruction:

    pip install -e .

### Dependencies

The code was written in Python 3.9. The packages `numpy`, `matplotlib`, `scipy`, and `numba` are required for using `pynewmarkdisp`. They should be installed along with the package, however, all of them can also be manually installed from the PyPI repository by opening a terminal and typing the following code lines:

    pip install numpy==1.22.4
    pip install matplotlib==3.7.1
    pip install scipy==1.9.0
    pip install numba==0.56.4

## Citation

To cite `pyNewmarkDisp` in publications, use the following temporary reference (until an associated article is published):

    Montoya-Araque, E. A., Montoya-Noguera, S., & Lopez-Caballero, F. (202X). An open-source application software for spatial prediction of permanent displacements in earthquake-induced landslides by the Newmark sliding block method: pyNewmarkDisp v0.1.0. url=https://github.com/eamontoyaa/pyNewmarkDisp

A BibTeX entry for LaTeX users is:

``` bibtex
@software{MontoyaAraque_etal_202X,
author = {Montoya-Araque, Exneyder A. and Montoya-Noguera, Silvana and Lopez-Caballero, Fernando},
title = {{An open-source application software for spatial prediction of permanent displacements in earthquake-induced landslides by the Newmark sliding block method: \texttt{pyNewmarkDisp}}},
version = {v0.1.0},
url = {https://github.com/eamontoyaa/pyNewmarkDisp},
year = {202X}
}
```

## License

`pyNewmarkDisp` is a free and open-source application sorfware licensed under the terms of the [MIT license](https://opensource.org/license/mit/). The following is a copy of the license:

{!LICENSE!}
