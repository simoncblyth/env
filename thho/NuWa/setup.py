"""

  Deploy into python in "develop" mode for convenience ...
  (NB setuptools must be present in the target python)

thho@thho-laptop:~/env/thho/NuWa$ python setup.py develop
running develop
running egg_info
creating dybthho.egg-info
writing dybthho.egg-info/PKG-INFO
writing top-level names to dybthho.egg-info/top_level.txt
writing dependency_links to dybthho.egg-info/dependency_links.txt
writing manifest file 'dybthho.egg-info/SOURCES.txt'
writing manifest file 'dybthho.egg-info/SOURCES.txt'
running build_ext
Creating /usr/local/dyb/1.0.0-rc01/external/Python/2.5.2/debian_x86_gcc4/lib/python2.5/site-packages/dybthho.egg-link (link to .)
Adding dybthho 0.0.1 to easy-install.pth file

Installed /home/thho/env/thho/NuWa
Processing dependencies for dybthho==0.0.1
Finished processing dependencies for dybthho==0.0.1
thho@thho-laptop:~/env/thho/NuWa$

   
"""

from setuptools import setup

setup(
    name='dybthho',
    version='0.0.1',
    author='Taihsiang',
    author_email = 'thho@hep1.phys.ntu.edu.tw',
    description = 'dybthho utilities ',
    license = 'None',
    packages = ['dybthho'],
    )
