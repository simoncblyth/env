
from setuptools import setup, find_packages

#
#   REMEMBER ... for version changes here to take effect you MUST rerun : 
#       cd ~/e/vdbi ; python setup.py develop
#   which propagates the change into the egg-info
#

setup(
    name='vdbi',
    version="0.1",
    description="Web interface to DBI tables",
    author="Simon Blyth",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "rum", 
        "RumAlchemy==0.3dev-20090708", 
        "tw.rum",
        "tw.dynforms",
        "ConfigObj",
        "ipython",
        ],
    entry_points="""
       [rum.repositoryfactory]
       vdbisqlalchemy = vdbi.rumalchemy:DbiSARepositoryFactory

       [rum.policy]
       vdbipolicy = vdbi.rum.policy:DbiPolicy

       [rum.viewfactory]
       vdbitoscawidgets = vdbi.tw.rum:DbiWidgetFactory

       [console_scripts]
       vdbi = vdbi.app.command:vdbi
       vdbi_scrape = vdbi.app.command:vdbi_scrape
       vdbi_transfer_statics = vdbi.app.command:vdbi_transfer_statics

       [toscawidgets.widgets]
       # Register your widgets so they can be collected by archive_tw_resources
       widgets = vdbi.tw.rum


"""
)



