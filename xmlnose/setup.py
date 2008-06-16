
from setuptools import setup

setup(
    name='XMLNose',
    version='0.1',
    author='Simon C Blyth',
    author_email = 'blyth@hep1.phys.ntu.edu.tw',
    description = 'Nose XML output plugin providing the report format needed for the Bitten Trac plugin',
    license = 'Public domain',
    packages = ['xmlnose'],
    entry_points = {
        'nose.plugins.0.10': [
            'xmlout = xmlnose:XmlOutput'
            ]
        }

    )
