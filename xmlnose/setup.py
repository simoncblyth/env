import sys

#try:
#    import ez_setup
#    ez_setup.use_setuptools()
#except ImportError:
#    pass
#
##
## is setuptools standard with py2.5 ?
##
from setuptools import setup

setup(
    name='XML output plugin',
    version='0.1',
    author='Simon C Blyth',
    author_email = 'blyth@hep1.phys.ntu.edu.tw',
    description = 'Nose XML output plugin',
    license = 'Public domain',
    py_modules = ['xmlplug'],
    entry_points = {
        'nose.plugins.0.10': [
            'xmlout = xmlplug:XmlOutput'
            ]
        }

    )
