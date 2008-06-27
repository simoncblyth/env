"""

  Deploy into python in "develop" mode for convenience ...
  (NB setuptools must be present in the target python)

   
"""

from setuptools import setup

setup(
    name='DybTest',
    version='0.0.1',
    author='Simon C Blyth',
    author_email = 'blyth@hep1.phys.ntu.edu.tw',
    description = 'Dyb Python Scratchpad ',
    license = 'None',
    packages = ['DybTest'],
    )
