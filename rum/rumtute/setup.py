
from setuptools import setup, find_packages

setup(
    name='rumtute',
    version="0.1",
    description="simple customisation of Rum tutorial to demonstrate/isolate bugs",
    author="Simon Blyth",
    packages=find_packages(),
    include_package_data=True,
    entry_points="""

       [rum.viewfactory]
       tutetoscawidgets = rumtute.tw.rum:TuteWidgetFactory


"""
)



