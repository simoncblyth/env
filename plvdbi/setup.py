try:
    from setuptools import setup, find_packages
except ImportError:
    from ez_setup import use_setuptools
    use_setuptools()
    from setuptools import setup, find_packages

setup(
    name='plvdbi',
    version='0.1',
    description='',
    author='',
    author_email='',
    url='',
    install_requires=[
       #"Pylons>=0.9.7",
        "SQLAlchemy>=0.5",
        "Genshi>=0.4",
        "AuthKit>=0.4.3,<=0.4.99",
    ],
    setup_requires=["PasteScript>=1.6.3"],
    packages=find_packages(exclude=['ez_setup']),
    include_package_data=True,
    test_suite='nose.collector',
    package_data={'plvdbi': ['i18n/*/LC_MESSAGES/*.mo']},
    #message_extractors={'plvdbi': [
    #        ('**.py', 'python', None),
    #        ('public/**', 'ignore', None)]},
    zip_safe=False,
    paster_plugins=['PasteScript', 'Pylons'],
    entry_points="""
    [paste.app_factory]
    main = plvdbi.config.middleware:make_app

    [paste.app_install]
    main = pylons.util:PylonsInstaller

    [toscawidgets.widgets]
    # Register your widgets so they can be collected by archive_tw_resources
    widgets = vdbi.rum.widgets



    """,
)
