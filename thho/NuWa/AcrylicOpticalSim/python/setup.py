"""
Date: 20090201
[thho@hep52 python]$ python setup.py develop
running develop
running egg_info
creating ThhoNuWa.egg-info
writing ThhoNuWa.egg-info/PKG-INFO
writing top-level names to ThhoNuWa.egg-info/top_level.txt
writing dependency_links to ThhoNuWa.egg-info/dependency_links.txt
writing manifest file 'ThhoNuWa.egg-info/SOURCES.txt'
reading manifest file 'ThhoNuWa.egg-info/SOURCES.txt'
writing manifest file 'ThhoNuWa.egg-info/SOURCES.txt'
running build_ext
Creating /data/dyb/NuWa/1.1.0/external/Python/2.5.2/slc4_amd64_gcc346/lib/python2.5/site-packages/ThhoNuWa.egg-link (link to .)
Removing ThhoNuWa 0.0.1 from easy-install.pth file
Adding ThhoNuWa 0.0.1 to easy-install.pth file

Installed /misc/home/hep/thho/env/thho/NuWa/AcrylicOpticalSim/python
Processing dependencies for ThhoNuWa==0.0.1
Finished processing dependencies for ThhoNuWa==0.0.1
[thho@hep52 python]$
   
"""

from setuptools import setup

setup(
    name='ThhoNuWa',
    version='0.0.1',
    author='Taihsiang',
    author_email = 'thho@hep1.phys.ntu.edu.tw',
    description = 'Thho Dayabay NuWa utilities',
    license = 'None',
    packages = ['AcrylicOpticalSim']
    )
