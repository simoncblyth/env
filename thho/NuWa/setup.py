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


20090201
[thho@hep52 NuWa]$ python setup.py develop
running develop
running egg_info
writing ThhoNuWa.egg-info/PKG-INFO
writing top-level names to ThhoNuWa.egg-info/top_level.txt
writing dependency_links to ThhoNuWa.egg-info/dependency_links.txt
writing manifest file 'ThhoNuWa.egg-info/SOURCES.txt'
running build_ext
Creating /data/dyb/NuWa/1.1.0/external/Python/2.5.2/slc4_amd64_gcc346/lib/python2.5/site-packages/ThhoNuWa.egg-link (link to .)
ThhoNuWa 0.0.1 is already the active version in easy-install.pth

Installed /misc/home/hep/thho/env/thho/NuWa
Processing dependencies for ThhoNuWa==0.0.1
Finished processing dependencies for ThhoNuWa==0.0.1
[thho@hep52 NuWa]$

   
"""

from setuptools import setup

setup(
    name='ThhoNuWa',
    version='0.0.1',
    author='Taihsiang',
    author_email = 'thho@hep1.phys.ntu.edu.tw',
    description = 'Thho Dayabay NuWa utilities',
    license = 'None',
    packages = ['dybthho']
    )
