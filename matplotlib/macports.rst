
Attempt to MPL via macports
===============================


OSX macports issue
--------------------

::

        simon:~ blyth$ ipython-2.6
        Python 2.6.7 (r267:88850, Mar  5 2012, 13:46:47) 
        Type "copyright", "credits" or "license" for more information.

        IPython 0.12 -- An enhanced Interactive Python.
        ?         -> Introduction and overview of IPython's features.
        %quickref -> Quick reference.
        help      -> Python's own help system.
        object?   -> Details about 'object', use 'object??' for extra details.

        In [1]: import matplotlib.pyplot as plt
        ---------------------------------------------------------------------------
        RuntimeError                              Traceback (most recent call last)
        RuntimeError: module compiled against ABI version 2000000 but this version of numpy is 1000009
        ---------------------------------------------------------------------------
        ImportError                               Traceback (most recent call last)




Reinstall
----------

::

        g4pb:pubs blyth$ port search matplotlib
        py-matplotlib @1.1.1 (python, graphics, math)
            matlab-like syntax for creating plots in python

        py-matplotlib-basemap @1.0.5 (python, graphics, math)
            matplotlib toolkit for plotting data on map projections

        py24-matplotlib @1.1.1 (python, graphics, math)
            matlab-like syntax for creating plots in python

        py24-matplotlib-basemap @1.0.5 (python, graphics, math)
            matplotlib toolkit for plotting data on map projections

        py25-matplotlib @1.1.1 (python, graphics, math)
            matlab-like syntax for creating plots in python

        py25-matplotlib-basemap @1.0.5 (python, graphics, math)
            matplotlib toolkit for plotting data on map projections

        py26-matplotlib @1.1.1 (python, graphics, math)
            matlab-like syntax for creating plots in python

        py26-matplotlib-basemap @1.0.5 (python, graphics, math)
            matplotlib toolkit for plotting data on map projections

        py27-matplotlib @1.1.1 (python, graphics, math)
            matlab-like syntax for creating plots in python

        py27-matplotlib-basemap @1.0.5 (python, graphics, math)
            matplotlib toolkit for plotting data on map projections

        Found 10 ports.
        g4pb:pubs blyth$ 
        g4pb:pubs blyth$ 
        g4pb:pubs blyth$ port info py26-matplotlib
        py26-matplotlib @1.1.1, Revision 2 (python, graphics, math)
        Variants:             cairo, ghostscript, gtk2, latex, qt4, [+]tkinter, universal, wxpython

        Description:          Matplotlib is a pure python plotting library with the goal of making publication quality plots using a syntax familiar to matlab users. The library uses numpy for
                              handling large data sets and supports a variety of output backends. This port provides variants for the different GUIs (gtk2, tkinter, wxpython).
        Homepage:             http://matplotlib.sourceforge.net

        Library Dependencies: python26, py26-dateutil, py26-tz, py26-numpy, py26-configobj, py26-pyobjc-cocoa, freetype, libpng, py26-tkinter
        Platforms:            darwin
        License:              PSF BSD
        Maintainers:          ram@macports.org, openmaintainer@macports.org




fails at close to last hurdle
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

        --->  Fetching archive for py26-matplotlib
        --->  Attempting to fetch py26-matplotlib-1.1.1_2+tkinter.darwin_9.ppc.tbz2 from http://packages.macports.org/py26-matplotlib
        --->  Attempting to fetch py26-matplotlib-1.1.1_2+tkinter.darwin_9.ppc.tbz2 from http://mse.uk.packages.macports.org/sites/packages.macports.org/py26-matplotlib
        --->  Attempting to fetch py26-matplotlib-1.1.1_2+tkinter.darwin_9.ppc.tbz2 from http://lil.fr.packages.macports.org/py26-matplotlib
        --->  Fetching distfiles for py26-matplotlib
        --->  Attempting to fetch matplotlib-1.1.1.tar.gz from http://nchc.dl.sourceforge.net/matplotlib
        --->  Verifying checksum(s) for py26-matplotlib
        --->  Extracting py26-matplotlib
        --->  Applying patches to py26-matplotlib
        --->  Configuring py26-matplotlib
        --->  Building py26-matplotlib
        Error: org.macports.build for port py26-matplotlib returned: command execution failed
        To report a bug, follow the instructions in the guide:
            http://guide.macports.org/#project.tickets
        Error: Processing of port py26-matplotlib failed
        g4pb:matplotlib blyth$ 
        g4pb:matplotlib blyth$ 


::

        g4pb:matplotlib blyth$ sudo port install -v py26-matplotlib
        Password:
        --->  Computing dependencies for py26-matplotlib
        --->  Building py26-matplotlib
        Error: org.macports.build for port py26-matplotlib returned: command execution failed
        Please see the log file for port py26-matplotlib for details:
            /opt/local/var/macports/logs/_opt_local_var_macports_sources_rsync.macports.org_release_ports_python_py-matplotlib/py26-matplotlib/main.log
        To report a bug, follow the instructions in the guide:
            http://guide.macports.org/#project.tickets
        Error: Processing of port py26-matplotlib failed
        g4pb:matplotlib blyth$ 


From the log, see that pytz related::

        :info:build               dateutil: present, version unknown
        :info:build Traceback (most recent call last):
        :info:build   File "setup.py", line 185, in <module>
        :info:build     provide_pytz = check_provide_pytz(hasdatetime)
        :info:build   File "/opt/local/var/macports/build/_opt_local_var_macports_sources_rsync.macports.org_release_ports_python_py-matplotlib/py26-matplotlib/work/matplotlib-1.1.1/setupext.py", line 474, in check_provide_pytz
        :info:build     if pytz.__version__.endswith('mpl'):
        :info:build AttributeError: 'module' object has no attribute '__version__'
        :info:build Command failed:  cd "/opt/local/var/macports/build/_opt_local_var_macports_sources_rsync.macports.org_release_ports_python_py-matplotlib/py26-matplotlib/work/matplotlib-1.1.1" && /opt/local/Library/Frameworks/Python.framework/Versions/2.6/bin/python2.6 setup.py --no-user-cfg build
        :info:build Exit code: 1
        :error:build org.macports.build for port py26-matplotlib returned: command execution failed
        :debug:build Error code: CHILDSTATUS 53087 1
        :debug:build Backtrace: command execution failed
            while executing
            "system -nice 0 $fullcmdstring"



