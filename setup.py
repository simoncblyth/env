"""
   placed into sys.path with one time
      cd $ENV_HOME ; sudo python setup.py develop

"""
from setuptools import setup, find_packages

setup(name='Env',
      version='0.1',
      packages = find_packages(exclude=['*.tests*']),  
      )
