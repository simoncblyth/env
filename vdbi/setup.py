
from setuptools import setup, find_packages

setup(
    name='vdbi',
    version="0.1",
    description="web interface to DBI tables",
    author="Simon Blyth",
    packages=find_packages(),
    include_package_data=True,
    entry_points="""
       [rum.repositoryfactory]
       vdbisqlalchemy = vdbi.rumalchemy:DbiSARepositoryFactory

       [rum.policy]
       vdbipolicy = vdbi.rum.policy:DbiPolicy

       [rum.viewfactory]
       vdbitoscawidgets = vdbi.tw.rum:DbiWidgetFactory

       [console_scripts]

"""
)



