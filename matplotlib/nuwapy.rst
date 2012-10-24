Into NuWa python
=================

::


        [blyth@belle7 DybDbi]$ easy_install matplotlib
        Searching for matplotlib
        Reading http://pypi.python.org/simple/matplotlib/
        Reading http://matplotlib.sourceforge.net
        Reading http://sourceforge.net/project/showfiles.php?group_id=80706&package_id=82474
        Reading http://sourceforge.net/project/showfiles.php?group_id=80706
        Reading https://sourceforge.net/project/showfiles.php?group_id=80706&package_id=82474
        Reading https://sourceforge.net/projects/matplotlib/files/matplotlib/matplotlib-0.99.1/
        Reading https://sourceforge.net/projects/matplotlib/files/matplotlib/matplotlib-0.99.3/
        Reading https://sourceforge.net/projects/matplotlib/files/matplotlib/matplotlib-1.0.1/
        Reading http://sourceforge.net/projects/matplotlib/files/matplotlib/matplotlib-1.1.0/
        Reading https://sourceforge.net/project/showfiles.php?group_id=80706&package_id=278194
        Reading https://sourceforge.net/projects/matplotlib/files/matplotlib/matplotlib-1.0
        Reading http://sourceforge.net/projects/matplotlib/files/matplotlib/matplotlib-1.1.1/
        Best match: matplotlib 1.1.1-notests
        Downloading http://sourceforge.net/projects/matplotlib/files/matplotlib/matplotlib-1.1.1/matplotlib-1.1.1_notests.tar.gz/download
        Processing download
        Running matplotlib-1.1.1/setup.py -q bdist_egg --dist-dir /tmp/easy_install-50rbZG/matplotlib-1.1.1/egg-dist-tmp-uQpDSK
        basedirlist is: ['/usr/local', '/usr']
        ============================================================================
        BUILDING MATPLOTLIB
                    matplotlib: 1.1.1
                        python: 2.7 (r27:82500, Feb 16 2011, 11:40:18)  [GCC 4.1.2
                                20080704 (Red Hat 4.1.2-46)]
                      platform: linux2

        REQUIRED DEPENDENCIES
                         numpy: no
                                * You must install numpy 1.4 or later to build
                                * matplotlib.
        error: Setup script exited with 1
        [blyth@belle7 DybDbi]$ 


So install numpy first, with `easy_install numpy` many warnings and even errors but it seems to complete.


::

        [blyth@belle7 ~]$ python -c "import numpy as np ; print np.version.version, np.get_include()  "
        1.6.2 /data1/env/local/dyb/external/Python/2.7/i686-slc5-gcc41-dbg/lib/python2.7/site-packages/numpy-1.6.2-py2.7-linux-i686.egg/numpy/core/include
        [blyth@belle7 ~]$ 


Check mysql_python on belle7
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~








