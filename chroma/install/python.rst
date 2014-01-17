Python issues
================


Locale
-------

On Mavericks 10.9, whilst trying to find how to run chroma tests::

    (chroma_env)delta:chroma blyth$ python setup.py --help-commands
    Traceback (most recent call last):
      File "setup.py", line 88, in <module>
        test_suite = 'nose.collector',
      File "/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/distutils/core.py", line 138, in setup
        ok = dist.parse_command_line()
      File "/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/setuptools/dist.py", line 250, in parse_command_line
        result = _Distribution.parse_command_line(self)
      File "/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/distutils/dist.py", line 464, in parse_command_line
        if self.handle_display_options(option_order):
      File "/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/setuptools/dist.py", line 611, in handle_display_options
        return _Distribution.handle_display_options(self, option_order)
      File "/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/distutils/dist.py", line 669, in handle_display_options
        self.print_commands()
      File "/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/setuptools/dist.py", line 371, in print_commands
        cmdclass = ep.load(False) # don't require extras, we're not running
      File "/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/pkg_resources.py", line 2029, in load
        entry = __import__(self.module_name, globals(),globals(), ['__name__'])
      File "/usr/local/env/chroma_env/lib/python2.7/site-packages/Sphinx-1.2-py2.7.egg/sphinx/setup_command.py", line 20, in <module>
        from sphinx.application import Sphinx
      File "/usr/local/env/chroma_env/lib/python2.7/site-packages/Sphinx-1.2-py2.7.egg/sphinx/application.py", line 22, in <module>
        from docutils.parsers.rst import convert_directive_function, \
      File "/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/docutils/parsers/rst/__init__.py", line 74, in <module>
        import docutils.statemachine
      File "/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/docutils/statemachine.py", line 113, in <module>
        from docutils import utils
      File "/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/docutils/utils/__init__.py", line 20, in <module>
        import docutils.io
      File "/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/docutils/io.py", line 18, in <module>
        from docutils.utils.error_reporting import locale_encoding, ErrorString, ErrorOutput
      File "/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/docutils/utils/error_reporting.py", line 47, in <module>
        locale_encoding = locale.getlocale()[1] or locale.getdefaultlocale()[1]
      File "/usr/local/env/chroma_env/lib/python2.7/locale.py", line 511, in getdefaultlocale
        return _parse_localename(localename)
      File "/usr/local/env/chroma_env/lib/python2.7/locale.py", line 443, in _parse_localename
        raise ValueError, 'unknown locale: %s' % localename
    ValueError: unknown locale: UTF-8
    (chroma_env)delta:chroma blyth$ 
    (chroma_env)delta:chroma blyth$ 


Perhaps resolved by adding to .bash_profile::

    # http://stackoverflow.com/questions/19961239/pelican-3-3-pelican-quickstart-error-valueerror-unknown-locale-utf-8
    export LC_ALL=en_US.UTF-8
    export LANG=en_US.UTF-8


help commands trys to install stuff
--------------------------------------

::

    (chroma_env)delta:chroma blyth$ python setup.py --help-commands
    Downloading http://pypi.python.org/packages/source/d/distribute/distribute-0.6.35.tar.gz
    Extracting in /var/folders/qm/1p5gh0x94l3b0xqc8dpr9yn40000gn/T/easy_install-BTruBO/PyUblas-2013.1/temp/tmpWmfpCf
    Now working in /var/folders/qm/1p5gh0x94l3b0xqc8dpr9yn40000gn/T/easy_install-BTruBO/PyUblas-2013.1/temp/tmpWmfpCf/distribute-0.6.35
    Building a Distribute egg in /private/var/folders/qm/1p5gh0x94l3b0xqc8dpr9yn40000gn/T/easy_install-BTruBO/PyUblas-2013.1
    /private/var/folders/qm/1p5gh0x94l3b0xqc8dpr9yn40000gn/T/easy_install-BTruBO/PyUblas-2013.1/distribute-0.6.35-py2.7.egg
    -------------------------------------------------------------------------
    Setuptools conflict detected.
    -------------------------------------------------------------------------
    When I imported setuptools, I did not get the distribute version of
    setuptools, which is troubling--this package really wants to be used
    with distribute rather than the old setuptools package. More than likely,
    you have both distribute and setuptools installed, which is bad.

    See this page for more information:
    http://wiki.tiker.net/DistributeVsSetuptools
    -------------------------------------------------------------------------
    I will continue after a short while, fingers crossed.
    Hit Ctrl-C now if you'd like to think about the situation.
    -------------------------------------------------------------------------
    ^CTraceback (most recent call last):
      File "setup.py", line 88, in <module>
        test_suite = 'nose.collector',
      File "/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/distutils/core.py", line 112, in setup
        _setup_distribution = dist = klass(attrs)
      File "/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/setuptools/dist.py", line 239, in __init__
        self.fetch_build_eggs(attrs.pop('setup_requires'))
      File "/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/setuptools/dist.py", line 263, in fetch_build_eggs
        parse_requirements(requires), installer=self.fetch_build_egg
      File "/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/pkg_resources.py", line 564, in resolve
        dist = best[req.key] = env.best_match(req, self, installer)
      File "/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/pkg_resources.py", line 802, in best_match
        return self.obtain(req, installer) # try and download/install
      File "/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/pkg_resources.py", line 814, in obtain
        return installer(requirement)
      File "/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/setuptools/dist.py", line 313, in fetch_build_egg
        return cmd.easy_install(req)
      File "/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/setuptools/command/easy_install.py", line 593, in easy_install
        return self.install_item(spec, dist.location, tmpdir, deps)
      File "/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/setuptools/command/easy_install.py", line 623, in install_item
        dists = self.install_eggs(spec, download, tmpdir)
      File "/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/setuptools/command/easy_install.py", line 809, in install_eggs
        return self.build_and_install(setup_script, setup_base)
      File "/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/setuptools/command/easy_install.py", line 1015, in build_and_install
        self.run_setup(setup_script, setup_base, args)
      File "/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/setuptools/command/easy_install.py", line 1000, in run_setup
        run_setup(setup_script, args)
      File "/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/setuptools/sandbox.py", line 50, in run_setup
        lambda: execfile(
      File "/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/setuptools/sandbox.py", line 100, in run
        return func()
      File "/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/setuptools/sandbox.py", line 52, in <lambda>
        {'__file__':setup_script, '__name__':'__main__'}
      File "setup.py", line 143, in <module>
        
      File "setup.py", line 29, in main
        geant4_libs = check_output(['geant4-config','--libs']).split()
      File "/var/folders/qm/1p5gh0x94l3b0xqc8dpr9yn40000gn/T/easy_install-BTruBO/PyUblas-2013.1/aksetup_helper.py", line 37, in <module>
      File "/var/folders/qm/1p5gh0x94l3b0xqc8dpr9yn40000gn/T/easy_install-BTruBO/PyUblas-2013.1/aksetup_helper.py", line 15, in count_down_delay
    KeyboardInterrupt
    (chroma_env)delta:chroma blyth
