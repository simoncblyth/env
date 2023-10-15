pyvista-dir(){  echo $HOME/miniconda3/pkgs/pyvista-0.25.3-py_0/site-packages/pyvista ; }
pyvista-cd(){   cd $(pyvista-dir); }
pyvista-vi(){   vi $BASH_SOURCE ; }
pyvista-env(){  elocal- ; }
pyvista-usage(){ cat << EOU


PyVista : VTK for Humans
==========================

.. goal is to make 3D visualization and analysis approachable to
domain-scientists so they can focus on the research questions at hand.


* https://www.pyvista.org
* https://github.com/pyvista
* https://docs.pyvista.org
* https://banesullivan.com/python-blog/000-intro-to-pyvista.html
* https://docs.pyvista.org/examples/index.html


* https://docs.pyvista.org/examples/02-plot/depth-peeling.html#sphx-glr-examples-02-plot-depth-peeling-py

  Depth peeling is a technique to correctly render translucent geometry. 



TODO : Locate GUI window navigation control in pyvista as guide for doing similar
-----------------------------------------------------------------------------------


Exit Speedup
--------------

# PYVISTA_KILL_DISPLAY envvar is observed to speedup exiting from ipython after pyvista plotting 
# see https://github.com/pyvista/pyvista/blob/main/pyvista/plotting/plotting.py
export PYVISTA_KILL_DISPLAY=1



VTK
-----

* https://gitlab.kitware.com/vtk/vtk/-/issues/17917#note_783584
* ~/env/graphics/vtk_/vwin.py 


Plotter Interactive Key Mappings
----------------------------------

* https://dev.pyvista.org/api/plotting/plotting.html

+-------------------------------------------------+---------------------------------------------------+
|     Key                                         | Action                                            |
+----------------------+--------------------------+---------------------------------------------------+
| Linux/Windows        |  Mac                     |                                                   |
+======================+==========================+===================================================+
| q                    |                          | Close the rendering window                        |
+----------------------+--------------------------+---------------------------------------------------+
| f                    |                          | Focus and zoom in on a point                      |
+----------------------+--------------------------+---------------------------------------------------+
| v                    |                          | Isometric camera view                             |
+----------------------+--------------------------+---------------------------------------------------+
| w                    |                          | Switch all datasets to a wireframe representation |
+----------------------+--------------------------+---------------------------------------------------+
| r                    |                          | Reset the camera to view all datasets             |
+----------------------+--------------------------+---------------------------------------------------+
| s                    |                          | Switch all datasets to a surface representation   |
+----------------------+--------------------------+---------------------------------------------------+
| shift+clck/mid-clck  |  shift+click             | Pan the rendering scene                           |
+----------------------+--------------------------+---------------------------------------------------+
| left-click           | cmd+click                | Rotate the rendering scene in 3D                  |
+----------------------+--------------------------+---------------------------------------------------+
| ctrl+click           |                          | Rotate the rendering scene in 2D (view-plane)     |
+----------------------+--------------------------+---------------------------------------------------+
| mouse-whl/right-clck |  ctl+click               | Continuously zoom the rendering scene             |
+----------------------+--------------------------+---------------------------------------------------+
| shift+s              |                          | Save a screenhsot (only on BackgroundPlotter)     |
+----------------------+--------------------------+---------------------------------------------------+
| shift+c              |                          | Enable interactive cell selection/picking         |
+----------------------+--------------------------+---------------------------------------------------+
| up/down              |                          | Zoom in and out                                   |
+----------------------+--------------------------+---------------------------------------------------+
| +/-                  |                          | Increase/decrease the point size and line widths  |
+----------------------+--------------------------+---------------------------------------------------+


::

    epsilon:plotting blyth$ pwd
    /Users/blyth/miniconda3/lib/python3.7/site-packages/pyvista/plotting
    epsilon:plotting blyth$ grep add_key_event *.py 
    picking.py:        self.add_key_event('c', _clear_path_event_watcher)
    picking.py:        self.add_key_event('c', _clear_g_path_event_watcher)
    picking.py:        self.add_key_event('c', lambda: self.remove_actor(name))
    plotting.py:    def add_key_event(self, key, callback):
    plotting.py:        self.add_key_event('q', self._prep_for_close) # Add no matter what
    plotting.py:        self.add_key_event('b', b_left_down_callback)
    plotting.py:        self.add_key_event('v', lambda: self.isometric_view_interactive())
    plotting.py:        self.add_key_event('f', self.fly_to_mouse_position)
    plotting.py:        self.add_key_event('C', lambda: self.enable_cell_picking())
    plotting.py:        self.add_key_event('Up', lambda: self.camera.Zoom(1.05))
    plotting.py:        self.add_key_event('Down', lambda: self.camera.Zoom(0.95))
    plotting.py:        self.add_key_event('plus', lambda: self.increment_point_size_and_line_width(1))
    plotting.py:        self.add_key_event('minus', lambda: self.increment_point_size_and_line_width(-1))
    epsilon:plotting blyth$ 


plotting.iren : vtk.vtkRenderWindowInteractor
-----------------------------------------------

::

    3758         else:  # Allow user to interact
    3759             self.iren = vtk.vtkRenderWindowInteractor()
    3760             self.iren.LightFollowCameraOff()
    3761             self.iren.SetDesiredUpdateRate(30.0)
    3762             self.iren.SetRenderWindow(self.ren_win)
    3763             self.enable_trackball_style()
    3764             self._observers = {}    # Map of events to observers of self.iren
    3765             self._add_observer("KeyPressEvent", self.key_press_event)
    3766             self.update_style()




callback with line widget
--------------------------

* https://docs.pyvista.org/examples/03-widgets/line-widget.html

VTK version
--------------

::

    In [5]: print(vtk.vtkVersion.GetVTKVersion()) 
    8.2.0


add_key_event
--------------

* https://docs.pyvista.org/api/plotting/_autosummary/pyvista.Plotter.add_key_event.html

Data Model Intro
-----------------

* https://docs.pyvista.org/user-guide/what-is-a-mesh.html#what-is-a-mesh

Attributes are data values that live on either the points or cells of a mesh.
In PyVista, we work with both point data and cell data and allow easy access to
data dictionaries to hold arrays for attributes that live either on all points
or on all cells of a mesh. These attributes can be accessed in a
dictionary-like attribute attached to any PyVista mesh accessible as one of the
following:

point_data
cell_data
field_data

::

    mesh.point_data['my point values'] = np.arange(mesh.n_points)
    mesh.plot(scalars='my point values', cpos=cpos, show_edges=True)





* https://docs.pyvista.org/user-guide/data_model.html?highlight=dataset




Versions
-----------

* https://pypi.org/project/pyvista/#history

* 0.25.3 : from Jun 6, 2020 

::

    In [2]: pv.__version__
    Out[2]: '0.25.3'


Considering an upgrade
-------------------------

Online docs/examples are very often not compatible with 0.25.3 so are interested to upgrade, 
but am cautious as there are heavy dependencies like vtk.


::

    source $HOME/.miniconda3_config

    epsilon:pyvista_ blyth$ conda info pyvista    ## this takes minutes to return... and lists loadsa versions


    pyvista 0.32.0 pyhd8ed1ab_0
    ---------------------------
    file name   : pyvista-0.32.0-pyhd8ed1ab_0.tar.bz2
    name        : pyvista
    version     : 0.32.0
    build string: pyhd8ed1ab_0
    build number: 0
    channel     : https://conda.anaconda.org/conda-forge/noarch
    size        : 1.2 MB
    arch        : None
    constrains  : ()
    license     : MIT
    license_family: MIT
    md5         : ec0a2bf108a29185d2daadfe798ba4a3
    noarch      : python
    package_type: noarch_python
    platform    : None
    sha256      : 0ee5ded688501f2e17e3cf84bbbf7ab31f16d184b773c3bfa006bda05845a4d7
    subdir      : noarch
    timestamp   : 1631390362301
    url         : https://conda.anaconda.org/conda-forge/noarch/pyvista-0.32.0-pyhd8ed1ab_0.tar.bz2
    dependencies:
        appdirs
        imageio >=2.5.0
        meshio >=4.0.3,<5.0
        numpy
        python >=3.5
        scooby >=0.5.1
        typing_extensions
        vtk

    pyvista 0.25.3 py_0
    -------------------
    file name   : pyvista-0.25.3-py_0.tar.bz2
    name        : pyvista
    version     : 0.25.3
    build string: py_0
    build number: 0
    channel     : https://conda.anaconda.org/conda-forge/noarch
    size        : 1.1 MB
    arch        : None
    constrains  : ()
    license     : MIT
    license_family: MIT
    md5         : 2c5ca8724091fd8b2ce189a1bf6aa292
    noarch      : python
    package_type: noarch_python
    platform    : None
    sha256      : c340a31dbf4168a7a9b8c34eb3caaaa29f795aee4595843e53adca7b5ba4809f
    subdir      : noarch
    timestamp   : 1591638925187
    url         : https://conda.anaconda.org/conda-forge/noarch/pyvista-0.25.3-py_0.tar.bz2
    dependencies:
        appdirs
        imageio >=2.5.0
        meshio >=4.0.3,<5.0
        numpy
        python >=3.5
        scooby >=0.5.1
        vtk



updating seems too disruptive : so many changes that are not proceeding
--------------------------------------------------------------------------

::

    epsilon:pyvista_ blyth$ conda update pyvista
    Collecting package metadata (current_repodata.json): done
    Solving environment: done

    ## Package Plan ##

      environment location: /Users/blyth/miniconda3

      added / updated specs:
        - pyvista


    The following packages will be downloaded:

        package                    |            build
        ---------------------------|-----------------
        afterimage-1.21            |    h4dd67e6_1003         685 KB  conda-forge
        appdirs-1.4.4              |     pyh9f0ad1d_0          13 KB  conda-forge
        appnope-0.1.2              |   py37hf985489_2          10 KB  conda-forge
        apptools-5.1.0             |     pyh44b312d_0         123 KB  conda-forge
        attrs-21.4.0               |     pyhd8ed1ab_0          49 KB  conda-forge
        awkward-1.7.0              |   py37hd8d24ac_0        11.5 MB  conda-forge
        backports.functools_lru_cache-1.6.4|     pyhd8ed1ab_0           9 KB  conda-forge
        brotlipy-0.7.0             |py37h271585c_1003         357 KB  conda-forge
        bzip2-1.0.8                |       h0d85af4_4         155 KB  conda-forge
        c-ares-1.18.1              |       h0d85af4_0         100 KB  conda-forge
        ca-certificates-2021.10.8  |       h033912b_0         139 KB  conda-forge
        certifi-2021.10.8          |   py37hf985489_1         145 KB  conda-forge
        cfitsio-3.470              |       h422484a_7         1.3 MB  conda-forge
        chardet-3.0.4              |py37h2987424_1008         170 KB  conda-forge
        charset-normalizer-2.0.10  |     pyhd8ed1ab_0          34 KB  conda-forge
        colorama-0.4.4             |     pyh9f0ad1d_0          18 KB  conda-forge
        conda-4.11.0               |   py37hf985489_0        16.9 MB  conda-forge
        conda-package-handling-1.7.3|   py37h271585c_1         1.6 MB  conda-forge
        cryptography-36.0.1        |   py37h5e77fcc_0         1.3 MB  conda-forge
        cycler-0.11.0              |     pyhd8ed1ab_0          10 KB  conda-forge
        decorator-5.1.1            |     pyhd8ed1ab_0          12 KB  conda-forge
        envisage-6.0.1             |     pyhd8ed1ab_0         171 KB  conda-forge
        expat-2.4.3                |       he49afe7_0         144 KB  conda-forge
        fastcache-1.1.0            |   py37h271585c_3          28 KB  conda-forge
        fontconfig-2.13.1          |    h10f422b_1005         271 KB  conda-forge
        freetype-2.10.4            |       h4cff582_1         890 KB  conda-forge
        future-0.18.2              |   py37hf985489_4         713 KB  conda-forge
        gettext-0.19.8.1           |    haf92f58_1004         3.2 MB  conda-forge
        glew-2.1.0                 |       h046ec9c_2         692 KB  conda-forge
        glib-2.58.3                |py37h7c187be_1004         3.1 MB  conda-forge
        gmp-6.2.1                  |       h2e338ed_0         774 KB  conda-forge
        gmpy2-2.1.2                |   py37h60f582e_0         160 KB  conda-forge
        hdf4-4.2.15                |       hefd3b78_3         885 KB  conda-forge
        hdf5-1.10.6                |nompi_h34ad4e8_1111         3.0 MB  conda-forge
        idna-3.3                   |     pyhd8ed1ab_0          55 KB  conda-forge
        importlib-metadata-4.10.1  |   py37hf985489_0          32 KB  conda-forge
        importlib_metadata-4.10.1  |       hd8ed1ab_0           4 KB  conda-forge
        importlib_resources-5.4.0  |     pyhd8ed1ab_0          21 KB  conda-forge
        jedi-0.18.1                |   py37hf985489_0         995 KB  conda-forge
        jpeg-9d                    |       hbcb3906_0         250 KB  conda-forge
        kiwisolver-1.3.2           |   py37h737db71_1          58 KB  conda-forge
        krb5-1.17.2                |       h60d9502_0         1.2 MB  conda-forge
        libblas-3.9.0              |       8_openblas          11 KB  conda-forge
        libcblas-3.9.0             |       8_openblas          11 KB  conda-forge
        libedit-3.1.20191231       |       h0678c8f_2         103 KB  conda-forge
        libev-4.33                 |       haf1e3a3_1          99 KB  conda-forge
        libffi-3.2.1               |    hb1e8313_1007          42 KB  conda-forge
        libgfortran-4.0.0          |7_5_0_h1a10cd1_23          19 KB  conda-forge
        libgfortran4-7.5.0         |      h1a10cd1_23         1.1 MB  conda-forge
        libiconv-1.16              |       haf1e3a3_0         1.3 MB  conda-forge
        liblapack-3.9.0            |       8_openblas          11 KB  conda-forge
        libnetcdf-4.7.4            |nompi_h9d8a93f_107         1.2 MB  conda-forge
        libnghttp2-1.43.0          |       h6f36284_1         867 KB  conda-forge
        libopenblas-0.3.12         |openmp_h63d9170_1         8.7 MB  conda-forge
        libpng-1.6.37              |       h7cec526_2         313 KB  conda-forge
        libpq-12.3                 |       h7049927_3         2.6 MB  conda-forge
        libssh2-1.10.0             |       h52ee1ee_2         221 KB  conda-forge
        libxml2-2.9.10             |       h2c6e4a5_2         1.2 MB  conda-forge
        libzlib-1.2.11             |    h9173be1_1013          58 KB  conda-forge
        llvm-openmp-12.0.1         |       hda6cdc1_1         287 KB  conda-forge
        lz4-c-1.9.2                |       hb1e8313_3         152 KB  conda-forge
        mpc-1.2.1                  |       hbb51d92_0         103 KB  conda-forge
        mpfr-4.1.0                 |       h0f52abe_1         400 KB  conda-forge
        mpmath-1.2.1               |     pyhd8ed1ab_0         437 KB  conda-forge
        multidict-5.2.0            |   py37h271585c_1          56 KB  conda-forge
        ncurses-6.2                |       h2e338ed_4         881 KB  conda-forge
        nspr-4.32                  |       hcd9eead_1         247 KB  conda-forge
        nss-3.74                   |       h31e2bf1_0         2.0 MB  conda-forge
        numpy-1.21.5               |   py37h3c8089f_0         5.8 MB  conda-forge
        olefile-0.46               |     pyh9f0ad1d_1          32 KB  conda-forge
        openssl-1.1.1l             |       h0d85af4_0         1.9 MB  conda-forge
        parso-0.8.3                |     pyhd8ed1ab_0          69 KB  conda-forge
        pcre-8.45                  |       he49afe7_0         220 KB  conda-forge
        pexpect-4.8.0              |     pyh9f0ad1d_2          47 KB  conda-forge
        pickleshare-0.7.5          |          py_1003           9 KB  conda-forge
        pillow-7.2.0               |   py37hfd78ece_0         624 KB  conda-forge
        pip-21.3.1                 |     pyhd8ed1ab_0         1.2 MB  conda-forge
        prompt-toolkit-3.0.24      |     pyha770c72_0         249 KB  conda-forge
        ptyprocess-0.7.0           |     pyhd3deb0d_0          16 KB  conda-forge
        pycosat-0.6.3              |py37h271585c_1009         114 KB  conda-forge
        pycparser-2.21             |     pyhd8ed1ab_0         100 KB  conda-forge
        pyface-7.3.0               |     pyh44b312d_1         723 KB  conda-forge
        pygments-2.11.2            |     pyhd8ed1ab_0         796 KB  conda-forge
        pyopenssl-21.0.0           |     pyhd8ed1ab_0          48 KB  conda-forge
        pyparsing-3.0.7            |     pyhd8ed1ab_0          79 KB  conda-forge
        pyqt-5.12.3                |   py37hf985489_8          22 KB  conda-forge
        pyqt-impl-5.12.3           |   py37hab5ec1f_8         4.7 MB  conda-forge
        pyqt5-sip-4.19.18          |   py37h070e122_8         274 KB  conda-forge
        pyqtchart-5.12             |   py37hab5ec1f_8         215 KB  conda-forge
        pyqtwebengine-5.12.1       |   py37hab5ec1f_8         140 KB  conda-forge
        pysocks-1.7.1              |   py37hf985489_4          28 KB  conda-forge
        pythia8-8.244              |   py37h54c7649_2        26.0 MB  conda-forge
        python-dateutil-2.8.2      |     pyhd8ed1ab_0         240 KB  conda-forge
        python-xxhash-2.0.2        |   py37h271585c_1          19 KB  conda-forge
        python_abi-3.7             |          2_cp37m           4 KB  conda-forge
        pyvista-0.33.2             |     pyhd8ed1ab_0         1.3 MB  conda-forge
        readline-8.1               |       h05e3726_0         266 KB  conda-forge
        requests-2.27.1            |     pyhd8ed1ab_0          53 KB  conda-forge
        ruamel_yaml-0.15.80        |py37h271585c_1006         242 KB  conda-forge
        scooby-0.5.11              |     pyhd8ed1ab_0          16 KB  conda-forge
        setuptools-60.5.0          |   py37hf985489_0        1022 KB  conda-forge
        six-1.16.0                 |     pyh6c4a22f_0          14 KB  conda-forge
        sqlite-3.37.0              |       h23a322b_0         1.8 MB  conda-forge
        tbb-2020.2                 |       h940c156_4         132 KB  conda-forge
        tk-8.6.11                  |       h5dbffcc_1         3.3 MB  conda-forge
        tornado-6.1                |   py37h271585c_2         644 KB  conda-forge
        tqdm-4.62.3                |     pyhd8ed1ab_0          80 KB  conda-forge
        traitlets-5.1.1            |     pyhd8ed1ab_0          82 KB  conda-forge
        traits-6.3.2               |   py37h271585c_0         5.0 MB  conda-forge
        traitsui-7.2.0             |     pyhd8ed1ab_0         930 KB  conda-forge
        typing-extensions-4.0.1    |       hd8ed1ab_0           8 KB  conda-forge
        typing_extensions-4.0.1    |     pyha770c72_0          26 KB  conda-forge
        uproot-base-4.1.8          |     pyhb877337_0         201 KB  conda-forge
        urllib3-1.26.8             |     pyhd8ed1ab_1         100 KB  conda-forge
        vdt-0.4.3                  |       h046ec9c_0          32 KB  conda-forge
        wcwidth-0.2.5              |     pyh9f0ad1d_2          33 KB  conda-forge
        wheel-0.37.1               |     pyhd8ed1ab_0          31 KB  conda-forge
        xz-5.2.5                   |       haf1e3a3_1         228 KB  conda-forge
        yaml-0.2.5                 |       h0d85af4_2          82 KB  conda-forge
        yarl-1.7.2                 |   py37h271585c_1         127 KB  conda-forge
        zipp-3.7.0                 |     pyhd8ed1ab_0          12 KB  conda-forge
        zlib-1.2.11                |    h9173be1_1013          87 KB  conda-forge
        zstandard-0.17.0           |   py37h271585c_0         743 KB  conda-forge
        ------------------------------------------------------------
                                               Total:       133.6 MB

    The following NEW packages will be INSTALLED:

      charset-normalizer conda-forge/noarch::charset-normalizer-2.0.10-pyhd8ed1ab_0
      colorama           conda-forge/noarch::colorama-0.4.4-pyh9f0ad1d_0
      importlib-metadata conda-forge/osx-64::importlib-metadata-4.10.1-py37hf985489_0
      importlib_metadata conda-forge/noarch::importlib_metadata-4.10.1-hd8ed1ab_0
      importlib_resourc~ conda-forge/noarch::importlib_resources-5.4.0-pyhd8ed1ab_0
      libgfortran4       conda-forge/osx-64::libgfortran4-7.5.0-h1a10cd1_23
      libzlib            conda-forge/osx-64::libzlib-1.2.11-h9173be1_1013
      pyqt-impl          conda-forge/osx-64::pyqt-impl-5.12.3-py37hab5ec1f_8
      pyqt5-sip          conda-forge/osx-64::pyqt5-sip-4.19.18-py37h070e122_8
      pyqtchart          conda-forge/osx-64::pyqtchart-5.12-py37hab5ec1f_8
      pyqtwebengine      conda-forge/osx-64::pyqtwebengine-5.12.1-py37hab5ec1f_8
      zipp               conda-forge/noarch::zipp-3.7.0-pyhd8ed1ab_0

    The following packages will be REMOVED:

      cachetools-4.2.1-pyhd8ed1ab_0
      cctools-949.0.1-h22b1bf0_7
      cftime-1.2.1-py37h5be27a9_0
      gfortran_impl_osx-64-7.5.0-h970e067_1
      gfortran_osx-64-7.5.0-hb7f2cba_5
      h5py-2.10.0-nompi_py37h28defc4_104
      ipython_genutils-0.2.0-py_1
      isl-0.22.1-hb1e8313_2
      lcms2-2.11-h174193d_0
      ld64-530-7
      libllvm9-9.0.1-h7475705_1
      meshio-4.0.16-py_0
      netcdf4-1.5.4-nompi_py37h386ecc7_100
      sip-4.19.8-py37h0a44026_1000
      uproot-methods-0.9.2-pyhd8ed1ab_0

    The following packages will be UPDATED:

      appdirs                                        1.4.3-py_1 --> 1.4.4-pyh9f0ad1d_0
      appnope                           0.1.0-py37hc8dfbb8_1001 --> 0.1.2-py37hf985489_2
      apptools                                       4.5.0-py_0 --> 5.1.0-pyh44b312d_0
      attrs                                 20.3.0-pyhd3deb0d_0 --> 21.4.0-pyhd8ed1ab_0
      awkward                              1.1.2-py37h9af6487_0 --> 1.7.0-py37hd8d24ac_0
      backports.functoo~                             1.6.1-py_0 --> 1.6.4-pyhd8ed1ab_0
      brotlipy                          0.7.0-py37h9bfed18_1000 --> 0.7.0-py37h271585c_1003
      bzip2                                    1.0.8-h0b31af3_2 --> 1.0.8-h0d85af4_4
      c-ares                                  1.16.1-haf1e3a3_0 --> 1.18.1-h0d85af4_0
      ca-certificates                      2020.12.5-h033912b_0 --> 2021.10.8-h033912b_0
      certifi                          2020.12.5-py37hf985489_1 --> 2021.10.8-py37hf985489_1
      cfitsio                                  3.470-hdf94aef_6 --> 3.470-h422484a_7
      chardet                           3.0.4-py37hc8dfbb8_1006 --> 3.0.4-py37h2987424_1008
      conda                                4.9.2-py37hf985489_0 --> 4.11.0-py37hf985489_0
      conda-package-han~                   1.6.0-py37h9bfed18_2 --> 1.7.3-py37h271585c_1
      cryptography                           3.0-py37h94e4008_0 --> 36.0.1-py37h5e77fcc_0
      cycler                                        0.10.0-py_2 --> 0.11.0-pyhd8ed1ab_0
      decorator                                      4.4.2-py_0 --> 5.1.1-pyhd8ed1ab_0
      envisage                                       4.9.2-py_0 --> 6.0.1-pyhd8ed1ab_0
      expat                                    2.2.9-h4a8c4bd_2 --> 2.4.3-he49afe7_0
      fastcache                            1.1.0-py37h9bfed18_1 --> 1.1.0-py37h271585c_3
      fontconfig                           2.13.1-h79c0d67_1002 --> 2.13.1-h10f422b_1005
      freetype                                2.10.2-h8da9a1a_0 --> 2.10.4-h4cff582_1
      future                              0.18.2-py37hc8dfbb8_1 --> 0.18.2-py37hf985489_4
      gettext                            0.19.8.1-h46ab8bc_1002 --> 0.19.8.1-haf92f58_1004
      glew                                     2.1.0-h4a8c4bd_0 --> 2.1.0-h046ec9c_2
      gmp                                      6.2.0-hb1e8313_2 --> 6.2.1-h2e338ed_0
      gmpy2                              2.1.0b1-py37h4160ff4_0 --> 2.1.2-py37h60f582e_0
      hdf4                                 4.2.13-h84186c3_1003 --> 4.2.15-hefd3b78_3
      hdf5                            1.10.6-nompi_haae91d6_101 --> 1.10.6-nompi_h34ad4e8_1111
      idna                                    2.10-pyh9f0ad1d_0 --> 3.3-pyhd8ed1ab_0
      jedi                                        0.15.2-py37_0 --> 0.18.1-py37hf985489_0
      kiwisolver                           1.2.0-py37ha1cc60f_0 --> 1.3.2-py37h737db71_1
      krb5                                    1.17.1-h14dd6a4_2 --> 1.17.2-h60d9502_0
      libblas                                 3.8.0-17_openblas --> 3.9.0-8_openblas
      libcblas                                3.8.0-17_openblas --> 3.9.0-8_openblas
      libedit                           3.1.20191231-hed1e85f_1 --> 3.1.20191231-h0678c8f_2
      libev                                     4.33-haf1e3a3_0 --> 4.33-haf1e3a3_1
      libgfortran                                       4.0.0-2 --> 4.0.0-7_5_0_h1a10cd1_23
      libiconv                               1.15-h0b31af3_1006 --> 1.16-haf1e3a3_0
      liblapack                               3.8.0-17_openblas --> 3.9.0-8_openblas
      libnetcdf                        4.7.4-nompi_hc5b2cf3_105 --> 4.7.4-nompi_h9d8a93f_107
      libnghttp2                              1.41.0-h8a08a2b_1 --> 1.43.0-h6f36284_1
      libopenblas                      0.3.10-openmp_h63d9170_4 --> 0.3.12-openmp_h63d9170_1
      libpng                                  1.6.37-hbbe82c9_1 --> 1.6.37-h7cec526_2
      libpq                                     12.3-h489d428_0 --> 12.3-h7049927_3
      libssh2                                  1.9.0-h39bdce6_5 --> 1.10.0-h52ee1ee_2
      llvm-openmp                             10.0.1-h28b9765_0 --> 12.0.1-hda6cdc1_1
      lz4-c                                    1.9.2-h4a8c4bd_1 --> 1.9.2-hb1e8313_3
      mpc                                   1.1.0-ha57cd0f_1009 --> 1.2.1-hbb51d92_0
      mpfr                                     4.0.2-h72d8aaf_1 --> 4.1.0-h0f52abe_1
      mpmath                                         1.1.0-py_0 --> 1.2.1-pyhd8ed1ab_0
      multidict                            4.7.5-py37h60d8a13_2 --> 5.2.0-py37h271585c_1
      ncurses                                    6.2-hb1e8313_1 --> 6.2-h2e338ed_4
      nspr                                   4.20-h0a44026_1000 --> 4.32-hcd9eead_1
      nss                                       3.47-hc0980d9_0 --> 3.74-h31e2bf1_0
      numpy                               1.19.1-py37h7e69742_0 --> 1.21.5-py37h3c8089f_0
      olefile                                         0.46-py_0 --> 0.46-pyh9f0ad1d_1
      openssl                                 1.1.1k-h0d85af4_0 --> 1.1.1l-h0d85af4_0
      parso                                  0.7.1-pyh9f0ad1d_0 --> 0.8.3-pyhd8ed1ab_0
      pcre                                      8.44-h4a8c4bd_0 --> 8.45-he49afe7_0
      pexpect            conda-forge/osx-64::pexpect-4.8.0-py3~ --> conda-forge/noarch::pexpect-4.8.0-pyh9f0ad1d_2
      pickleshare        conda-forge/osx-64::pickleshare-0.7.5~ --> conda-forge/noarch::pickleshare-0.7.5-py_1003
      pip                                           20.2.1-py_0 --> 21.3.1-pyhd8ed1ab_0
      prompt-toolkit                                 3.0.5-py_1 --> 3.0.24-pyha770c72_0
      ptyprocess                                  0.6.0-py_1001 --> 0.7.0-pyhd3deb0d_0
      pycosat                           0.6.3-py37h9bfed18_1004 --> 0.6.3-py37h271585c_1009
      pycparser                               2.20-pyh9f0ad1d_2 --> 2.21-pyhd8ed1ab_0
      pyface                                 7.0.0-pyh9f0ad1d_1 --> 7.3.0-pyh44b312d_1
      pygments                                       2.6.1-py_0 --> 2.11.2-pyhd8ed1ab_0
      pyopenssl                                     19.1.0-py_1 --> 21.0.0-pyhd8ed1ab_0
      pyparsing                              2.4.7-pyh9f0ad1d_0 --> 3.0.7-pyhd8ed1ab_0
      pyqt                                5.12.3-py37ha62fc16_3 --> 5.12.3-py37hf985489_8
      pysocks                              1.7.1-py37hc8dfbb8_1 --> 1.7.1-py37hf985489_4
      python-dateutil                                2.8.1-py_0 --> 2.8.2-pyhd8ed1ab_0
      python-xxhash                        2.0.0-py37hf967b71_1 --> 2.0.2-py37h271585c_1
      python_abi                                    3.7-1_cp37m --> 3.7-2_cp37m
      pyvista                                       0.25.3-py_0 --> 0.33.2-pyhd8ed1ab_0
      readline                                   8.0-h0678c8f_2 --> 8.1-h05e3726_0
      requests                              2.24.0-pyh9f0ad1d_0 --> 2.27.1-pyhd8ed1ab_0
      ruamel_yaml                     0.15.80-py37h9bfed18_1001 --> 0.15.80-py37h271585c_1006
      scooby                                 0.5.6-pyh9f0ad1d_0 --> 0.5.11-pyhd8ed1ab_0
      setuptools                          49.2.1-py37hc8dfbb8_0 --> 60.5.0-py37hf985489_0
      six                                   1.15.0-pyh9f0ad1d_0 --> 1.16.0-pyh6c4a22f_0
      sqlite                                  3.35.2-h44b9ce1_0 --> 3.37.0-h23a322b_0
      tbb                                     2019.9-ha1b3eb9_1 --> 2020.2-h940c156_4
      tk                                      8.6.10-hbbe82c9_0 --> 8.6.11-h5dbffcc_1
      tornado                              6.0.4-py37h9bfed18_1 --> 6.1-py37h271585c_2
      tqdm                                  4.48.2-pyh9f0ad1d_0 --> 4.62.3-pyhd8ed1ab_0
      traitlets          conda-forge/osx-64::traitlets-4.3.3-p~ --> conda-forge/noarch::traitlets-5.1.1-pyhd8ed1ab_0
      traits                               6.1.0-py37h9bfed18_0 --> 6.3.2-py37h271585c_0
      traitsui                               7.0.1-pyh9f0ad1d_0 --> 7.2.0-pyhd8ed1ab_0
      typing-extensions                               3.7.4.3-0 --> 4.0.1-hd8ed1ab_0
      typing_extensions                            3.7.4.3-py_0 --> 4.0.1-pyha770c72_0
      uproot-base                            4.0.6-pyh985f01a_0 --> 4.1.8-pyhb877337_0
      urllib3                                      1.25.10-py_0 --> 1.26.8-pyhd8ed1ab_1
      wcwidth                                0.2.5-pyh9f0ad1d_1 --> 0.2.5-pyh9f0ad1d_2
      wheel                                         0.34.2-py_1 --> 0.37.1-pyhd8ed1ab_0
      yaml                                     0.2.5-h0b31af3_0 --> 0.2.5-h0d85af4_2
      yarl                                 1.6.3-py37h4b544eb_0 --> 1.7.2-py37h271585c_1
      zlib                                 1.2.11-h0b31af3_1006 --> 1.2.11-h9173be1_1013
      zstandard                           0.15.2-py37h271585c_0 --> 0.17.0-py37h271585c_0

    The following packages will be DOWNGRADED:

      afterimage                             1.21-hf755657_1003 --> 1.21-h4dd67e6_1003
      glib                                    2.65.0-h577aef8_0 --> 2.58.3-py37h7c187be_1004
      jpeg                                        9d-h0b31af3_0 --> 9d-hbcb3906_0
      libffi                                3.2.1-h4a8c4bd_1007 --> 3.2.1-hb1e8313_1007
      libxml2                                 2.9.10-h7fdee97_2 --> 2.9.10-h2c6e4a5_2
      pillow                               7.2.0-py37hfd78ece_1 --> 7.2.0-py37hfd78ece_0
      pythia8                              8.244-py37hdadc0f0_2 --> 8.244-py37h54c7649_2
      vdt                                      0.4.3-h6de7cb9_0 --> 0.4.3-h046ec9c_0
      xz                                       5.2.5-h0b31af3_1 --> 5.2.5-haf1e3a3_1


    Proceed ([y]/n)? n


    CondaSystemExit: Exiting.

    epsilon:pyvista_ blyth$ 






Others
--------

* https://yt-project.org/docs/dev/visualizing/plots.html


Precise Control of Viewpoint/Camera to match a render ?
----------------------------------------------------------

* manage to get same camera point and focus as the render but the 
  field of view is very different 


* https://vtk.org/doc/nightly/html/classvtkCamera.html

* https://blenderartists.org/t/whats-the-difference-between-orthographic-view-and-isometric-view/1167101/2

* https://github.com/Kitware/VTK/blob/master/Wrapping/Python/README.md




    epsilon:plotting blyth$ pwd
    /Users/blyth/miniconda3/lib/python3.7/site-packages/pyvista/plotting

    epsilon:plotting blyth$ grep camera.Set *.py 
    background_renderer.py:        self.camera.SetFocalPoint(xc, yc, 0.0)
    background_renderer.py:        self.camera.SetPosition(xc, yc, d)
    background_renderer.py:        self.camera.SetParallelScale(0.5 * yd / self._scale)
    plotting.py:        self.camera.SetThickness(path.length)
    renderer.py:            self.camera.SetPosition(scale_point(self.camera, camera_location[0],
    renderer.py:            self.camera.SetFocalPoint(scale_point(self.camera, camera_location[1],
    renderer.py:            self.camera.SetViewUp(camera_location[2])
    renderer.py:        self.camera.SetFocalPoint(scale_point(self.camera, point, invert=False))
    renderer.py:        self.camera.SetPosition(scale_point(self.camera, point, invert=False))
    renderer.py:        self.camera.SetViewUp(vector)
    renderer.py:        self.camera.SetParallelProjection(True)
    renderer.py:        self.camera.SetParallelProjection(False)
    renderer.py:        self.camera.SetModelTransformMatrix(transform.GetMatrix())
    epsilon:plotting blyth$ 



::

    In [3]: pl.camera                                                                                                                                                                                        
    Out[3]: (vtkRenderingOpenGL2Python.vtkOpenGLCamera)0x16948fde0

    In [4]: dir(pl.camera)                                                                                                                                                                                   
    Out[4]: 
    ['AddObserver',
     'ApplyTransform',
     'Azimuth',
     'BreakOnError',
     'ComputeViewPlaneNormal',
     'DebugOff',
     'DebugOn',
     'DeepCopy',
     'Dolly',
     'Elevation',
     'FastDelete',
     'GetAddressAsString',
     'GetCameraLightTransformMatrix',
     'GetClassName',
     'GetClippingRange',
     'GetCommand',
     'GetCompositeProjectionTransformMatrix',
     'GetDebug',
     'GetDirectionOfProjection',
     'GetDistance',
     'GetExplicitProjectionTransformMatrix',
     'GetEyeAngle',
     'GetEyePlaneNormal',
     'GetEyePosition',
     'GetEyeSeparation',
     'GetEyeTransformMatrix',


    In [39]: pl.camera.GetParallelScale()                                                                                                                                                                    
    Out[39]: 1021.7520079765083




    In [30]: m = pl.camera.GetViewTransformMatrix()                                                                                                                                                          
    In [31]: print(str(m))                                                                                                                                                                                   
    vtkMatrix4x4 (0x7fbd48f2cb30)
      Debug: Off
      Modified Time: 27087
      Reference Count: 2
      Registered Events: (none)
      Elements:
        0.707107 -0.707107 0 0 
        0.408248 0.408248 0.816497 0 
        -0.57735 -0.57735 0.57735 -3018.96 
        0 0 0 1 

    In [32]: m = pl.camera.GetEyeTransformMatrix()                                                                                                                                                           

    In [33]: print(str(m))                                                                                                                                                                                   
    vtkMatrix4x4 (0x7fbd48f2bdc0)
      Debug: Off
      Modified Time: 1182
      Reference Count: 2
      Registered Events: (none)
      Elements:
        1 0 0 0 
        0 1 0 0 
        0 0 1 0 
        0 0 0 1 







* https://vtk.org/doc/nightly/html/classvtkCamera.html

* https://vtk.org/doc/nightly/html/classvtkRenderer.html

* https://vtk.org/doc/nightly/html/classvtkRenderer.html#ae8055043e676defbbacff6f1ea65ad1e



    0100 class Renderer(vtkRenderer):
    0101     """Renderer class."""
    0102 
    ....
    1260     def reset_camera(self):
    1261         """Reset the camera of the active render window.
    1262 
    1263         The camera slides along the vector defined from camera position to focal point
    1264         until all of the actors can be seen.
    1265 
    1266         """
    1267         self.ResetCamera()
    1268         self.parent.render()
    1269         self.Modified()
    1270 
    ....
    1279     def view_isometric(self, negative=False):
    1280         """Reset the camera to a default isometric view.
    1281 
    1282         The view will show all the actors in the scene.
    1283 
    1284         """
    1285         self.camera_position = CameraPosition(*self.get_default_cam_pos(negative=negative))
    1286         self.camera_set = False
    1287         return self.reset_camera()


    1198     def set_scale(self, xscale=None, yscale=None, zscale=None, reset_camera=True):
    1199         """Scale all the datasets in the scene.
    1200 
    1201         Scaling in performed independently on the X, Y and Z axis.
    1202         A scale of zero is illegal and will be replaced with one.
    1203 
    1204         """
    1205         if xscale is None:
    1206             xscale = self.scale[0]
    1207         if yscale is None:
    1208             yscale = self.scale[1]
    1209         if zscale is None:
    1210             zscale = self.scale[2]
    1211         self.scale = [xscale, yscale, zscale]
    1212 
    1213         # Update the camera's coordinate system
    1214         transform = vtk.vtkTransform()
    1215         transform.Scale(xscale, yscale, zscale)
    1216         self.camera.SetModelTransformMatrix(transform.GetMatrix())
    1217         self.parent.render()
    1218         if reset_camera:
    1219             self.update_bounds_axes()
    1220             self.reset_camera()
    1221         self.Modified()




::


    In [4]: pl = pv.Plotter()                                                                                                                                                                                

    In [5]: pos = hposi[:,:3]                                                                                                                                                                                

    In [6]: pl.add_points(pos)                                                                                                                                                                               
    Out[6]: (vtkRenderingOpenGL2Python.vtkOpenGLActor)0x17251a9f0


    In [8]: pl.view_vector??                                                                                                                                                                                 
    Signature: pl.view_vector(vector, viewup=None)
    Source:   
        def view_vector(self, vector, viewup=None):
            """Point the camera in the direction of the given vector."""
            focal_pt = self.center
            if viewup is None:
                viewup = rcParams['camera']['viewup']
            cpos = CameraPosition(vector + np.array(focal_pt),
                    focal_pt, viewup)
            self.camera_position = cpos
            return self.reset_camera()
    File:      ~/miniconda3/lib/python3.7/site-packages/pyvista/plotting/renderer.py
    Type:      method




    pl.show?

    cpos : list(tuple(floats))
        The camera position to use

    height : int, optional
        height for panel pane. Only used with panel.

    Return
    ------
    cpos : list
        List of camera position, focal point, and view up



    In [11]: pl.camera_position                                                                                                                                                                              
    Out[11]: 
    [(-20.177947998046875, -20.16827392578125, 3897.8118510121158),
     (-20.177947998046875, -20.16827392578125, 22.314544677734375),
     (0.0, 1.0, 0.0)]

    In [12]: type(pl.camera_position)                                                                                                                                                                        
    Out[12]: pyvista.plotting.renderer.CameraPosition


    In [18]: from pyvista.plotting.renderer import CameraPosition as CP                                                                                                                                      

    In [19]: CP?                                                                                                                                                                                             
    Init signature: CP(position, focal_point, viewup)
    Docstring:      Container to hold camera location attributes.
    Init docstring: Initialize a new camera position descriptor.
    File:           ~/miniconda3/lib/python3.7/site-packages/pyvista/plotting/renderer.py
    Type:           type
    Subclasses:     

    In [20]: CP??                 









Comparisons with other viz tools
---------------------------------

* https://github.com/pyvista/pyvista/issues/146

* https://github.com/marcomusy/vedo

* https://github.com/vispy/vispy

  Direct to OpenGL 


* https://yt-project.org


PyVista vs Mayavi
--------------------

* https://github.com/pyvista/pyvista/issues/146

You're definitely right that there is a lot of overlap in features between
Mayavi and pyvista! I do however think pyvista is approaching 3D viz in a
totally different way the Mayavi... first, pyvista is truly an interface to the
Visualization Toolkit. We provide an easy to use interface to VTK's Python
bindings making accessing VTK data objects simple and fast. This allows pyvista
to merge into any existing Python VTK code as pyvista objects are instances of
VTK objects. It also stays true to VTK's object-oriented approach.

For example, in pyvista we simply wrap common VTK classes with properties and
methods to make accessing the underlying data within the VTK data object and
using VTK filters more intuitive so that users don't need to know the nuances
of creating VTK pipelines and remember all the different VTK classes for
filters, etc. I think Mayavi has a similar effort in this regard but I don't
know enough to comment too much further. I do know that the differences bewteen
how pyvista and Mayavi make VTK filters available to the user are stark:



PyVistaQt
-----------

* https://github.com/pyvista/pyvistaqt
* http://qtdocs.pyvista.org/usage.html

The python package pyvistaqt extends the functionality of pyvista through the
usage of PyQt5. Since PyQt5 operates in a separate thread than VTK, you can
similtaniously have an active VTK plot and a non-blocking Python session.


::

    In [3]: from pyvistaqt import BackgroundPlotter
    In [4]: pl = BackgroundPlotter()
    In [5]: pl.add_mesh(mesh) 


PyVista window interaction 
----------------------------

* https://docs.pyvista.org/plotting/plotting.html#plotting-ref


Model : mesh, cells, nodes, attributes
-----------------------------------------

* https://docs.pyvista.org/getting-started/what-is-a-mesh.html 

Cells aren’t limited to voxels, they could be a triangle between three nodes, a
line between two nodes, or even a single node could be its own cell (but that’s
a special case).

Attributes are data values that live on either the nodes or cells of a mesh


Attributes
~~~~~~~~~~~~

::

    In [75]: mesh.point_arrays                                                                                                                                               
    Out[75]: 
    pyvista DataSetAttributes
    Association: POINT
    Contains keys:
        sample_point_scalars
        VTKorigID

    In [77]: mesh.point_arrays['sample_point_scalars']                                                                                                                       
    Out[77]: 
    pyvista_ndarray([  1,   2,   4,   6,   8,  10,  12,  15,  19,  22,  23,
                      25,  27,  29,  31,  33,  36,  40,  44,  46,  48,  50,
                      52,  54,  56,  58,  60,  63,  65,  67,  69,  71,  73,
                      75,  77,  79,  91,  93,  95,  97,  99, 101, 103, 105,
                     107, 119, 121, 123, 125, 127, 129, 131, 133, 135, 147,
                     149, 151, 153, 155, 157, 159, 161, 163, 175, 177, 179,
                     181, 183, 185, 187, 189, 191, 203, 205, 207, 209, 211,
                     213, 215, 217, 219, 240, 242, 244, 246, 248, 250, 252,
                     254, 256, 286, 288, 290, 292, 294, 296, 298, 300, 302])

    In [79]: mesh.point_arrays['VTKorigID']                                                                                                                                  
    Out[79]: 
    pyvista_ndarray([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13,
                     14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
                     28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
                     42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
                     56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
                     70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83,
                     84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97,
                     98])


    In [76]: mesh.cell_arrays                                                                                                                                                
    Out[76]: 
    pyvista DataSetAttributes
    Association: CELL
    Contains keys:
        sample_cell_scalars


    In [78]: mesh.cell_arrays['sample_cell_scalars']                                                                                                                         
    Out[78]: 
    pyvista_ndarray([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14,
                     15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
                     29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40],
                    dtype=int32)

    In [79]:                                    



    In [80]: mesh.point_arrays['my point values'] = np.arange(mesh.n_points)                                                                                                

    In [81]: mesh.plot(scalars='my point values', cpos=bcpos, 
        ...:           show_edges=True, screenshot='beam_point_data.png')                                                                                                   



BackgroundPlotter moved to https://github.com/pyvista/pyvistaqt
------------------------------------------------------------------

::

    In [119]: plotter = pv.BackgroundPlotter()                                                                                                                              
    ---------------------------------------------------------------------------
    QtDeprecationError                        Traceback (most recent call last)
    <ipython-input-119-1a7f685be6b6> in <module>
    ----> 1 plotter = pv.BackgroundPlotter()

    ~/miniconda3/lib/python3.7/site-packages/pyvista/plotting/__init__.py in __init__(self, *args, **kwargs)
         33     def __init__(self, *args, **kwargs):
         34         """Empty init."""
    ---> 35         raise QtDeprecationError('BackgroundPlotter')
         36 
         37 

    QtDeprecationError: `BackgroundPlotter` has moved to pyvistaqt.
        You can install this from PyPI with: `pip install pyvistaqt`
        See https://github.com/pyvista/pyvistaqt






Examples
---------

::

    In [1]: from pyvista import examples 
    In [2]: dir(examples)          



EOU
}


pyvista-gr(){ pyvista-cd ; find . -name '*.py' -exec grep -H "${1:-UnstructuredGrid}" {} \; ; }
pyvista-gl(){ pyvista-cd ; find . -name '*.py' -exec grep -l "${1:-UnstructuredGrid}" {} \; ; }
