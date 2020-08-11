# === func-gen- : python/condaforge fgp python/condaforge.bash fgn condaforge fgh python src base/func.bash
condaforge-source(){   echo ${BASH_SOURCE} ; }
condaforge-edir(){ echo $(dirname $(condaforge-source)) ; }
condaforge-ecd(){  cd $(condaforge-edir); }
condaforge-dir(){  echo $LOCAL_BASE/env/python/condaforge ; }
condaforge-cd(){   cd $(condaforge-dir); }
condaforge-vi(){   vi $(condaforge-source) ; }
condaforge-env(){  elocal- ; }
condaforge-usage(){ cat << EOU

Conda Forge
=============

A community-led collection of recipes, build infrastructure and distributions for the conda package manager.

* https://conda-forge.org


conda config check prior to any changes
------------------------------------------

::

    con(base) epsilon:env blyth$ conda config --show channel_priority 
    channel_priority: flexible

    (base) epsilon:env blyth$ conda config --show channels
    channels:
      - defaults


Setting up conda-forge with anaconda3
----------------------------------------

https://conda-forge.org/docs/user/introduction.html#how-can-i-install-packages-from-conda-forge

::

    (base) epsilon:env blyth$ conda config --add channels conda-forge
    (base) epsilon:env blyth$ conda config --set channel_priority strict
    (base) epsilon:env blyth$ conda config --show channels
    channels:
      - conda-forge
      - defaults
    (base) epsilon:env blyth$ conda config --show channel_priority 
    channel_priority: strict
    (base) epsilon:env blyth$ 



Doing for miniconda3
----------------------

Seems miniconda3 shares the config of anaconda3::

    (base) epsilon:1 blyth$ conda config --add channels conda-forge
    Warning: 'conda-forge' already in 'channels' list, moving to the top
    (base) epsilon:1 blyth$ 






* https://conda-forge.org/docs/user/introduction.html


* https://anaconda.org/search?q=mayavi
* https://anaconda.org/conda-forge/mayavi

::

   conda install -c conda-forge mayavi 


After hours dumped megabytes looking like the below::

    Found conflicts! Looking for incompatible packages.
    This can take several minutes.  Press CTRL-C to abort.
    failed                                                                                                                                                                                            \  

    UnsatisfiableError: The following specifications were found
    to be incompatible with the existing python installation in your environment:

    Specifications:


::

    (base) epsilon:1 blyth$ conda install -c conda-forge mayavi -v -v -v 




      - alabaster -> python[version='2.7.*|3.5.*|3.6.*|>=2.7,<2.8.0a0|>=3.5,<3.6.0a0|>=3.6,<3.7.0a0|3.4.*|>=3.7,<3.8.0a0']
      - anaconda-navigator -> python[version='2.7.*|3.5.*|3.6.*|3.4.*']
      - anaconda-project -> python[version='2.7.*|3.5.*|3.6.*|>=3.5,<3.6.0a0|>=2.7,<2.8.0a0|>=3.7,<3.8.0a0|>=3.6,<3.7.0a0']
      - anaconda==2020.07 -> python[version='3.6.10|3.7.7|3.8.3',build='hf48f09d_4|hf48f09d_2|h26836e1_2']
      - atomicwrites -> python[version='2.7.*|3.5.*|3.6.*|>=2.7,<2.8.0a0|>=3.6,<3.7.0a0|>=3.5,<3.6.0a0|>=3.7,<3.8.0a0']
      - attrs -> python[version='2.7.*|3.5.*|3.6.*|>=2.7,<2.8.0a0|>=3.6,<3.7.0a0|>=3.7,<3.8.0a0|>=3.5,<3.6.0a0']
      - autopep8 -> python[version='2.7.*|3.5.*|3.6.*|3.4.*|>=2.7,<2.8.0a0|>=3.6,<3.7.0a0|>=3.7,<3.8.0a0|>=3.5,<3.6.0a0']
      - babel -> python[version='2.7.*|3.5.*|3.6.*|>=3.5,<3.6.0a0|>=3.6,<3.7.0a0|3.4.*|>=3.7,<3.8.0a0|>=2.7,<2.8.0a0']
      - backports -> python[version='2.7.*|3.5.*|3.6.*|>=3.5,<3.6.0a0|>=2.7,<2.8.0a0|>=3.7,<3.8.0a0|>=3.6,<3.7.0a0']
      - backports.functools_lru_cache -> python[version='2.7.*|3.5.*|3.6.*|3.4.*|>=2.7,<2.8.0a0']
      - bkcharts -> python=3.4
      - bleach -> python[version='2.7.*|3.5.*|3.6.*|3.4.*|>=2.7,<2.8.0a0|>=3.6,<3.7.0a0|>=3.7,<3.8.0a0|>=3.5,<3.6.0a0']
      - brotlipy -> python[version='3.4.*|3.6.9|3.8.*|3.7.*',build='0_73_pypy|1_73_pypy|2_73_pypy']
      - cffi -> python[version='2.7.*|3.5.*|3.6.*|3.6.9|3.6.9|3.6.9|>=3.6,<3.7.0a0|>=3.7,<3.8.0a0|>=3.8,<3.9.0a0|>=2.7,<2.8.0a0|>=3.5,<3.6.0a0|3.4.*',build='0_73_pypy|1_73_pypy|2_73_pypy']
      - colorama -> python[version='2.7.*|3.5.*|3.6.*|3.4.*|>=3.6,<3.7.0a0|>=2.7,<2.8.0a0|>=3.7,<3.8.0a0|>=3.5,<3.6.0a0']
      - conda-env -> python[version='2.7.*|3.4.*|3.5.*']
      - contextlib2 -> python[version='2.7.*|3.5.*|3.6.*|3.4.*|>=3.7,<3.8.0a0|>=3.6,<3.7.0a0|>=3.5,<3.6.0a0|>=2.7,<2.8.0a0']
      - cryptography -> python[version='3.4.*|3.6.9|3.7.*|3.8.*|<=3.3',build='0_73_pypy|1_73_pypy|2_73_pypy']
      - decorator -> python[version='2.7.*|3.5.*|3.6.*|3.4.*|>=2.7,<2.8.0a0|>=3.7,<3.8.0a0|>=3.6,<3.7.0a0|>=3.5,<3.6.0a0']
      - defusedxml -> python[version='2.7.*|3.5.*|3.6.*|3.4.*|>=2.7,<2.8.0a0|>=3.5,<3.6.0a0|>=3.6,<3.7.0a0|>=3.7,<3.8.0a0']
      - diff-match-patch -> python[version='2.7.*|3.4.*|3.5.*|3.6.*']
      - filelock -> python[version='2.7.*|3.5.*|3.6.*|3.4.*|>=3.6,<3.7.0a0|>=3.7,<3.8.0a0|>=2.7,<2.8.0a0|>=3.5,<3.6.0a0']
      - flask -> python[version='2.7.*|3.5.*|3.6.*|>=2.7,<2.8.0a0|>=3.5,<3.6.0a0|>=3.6,<3.7.0a0|3.4.*|>=3.7,<3.8.0a0']
      - gevent -> python[version='3.4.*|3.6.9|3.7.*|3.8.*',build='0_73_pypy|1_73_pypy|2_73_pypy']
      - glob2 -> python[version='2.7.*|3.4.*|3.5.*|3.6.*|>=2.7,<2.8.0a0|>=3.7,<3.8.0a0|>=3.6,<3.7.0a0|>=3.5,<3.6.0a0']
      - html5lib -> python[version='2.7.*|3.5.*|3.6.*|>=3.5,<3.6.0a0|>=2.7,<2.8.0a0|>=3.7,<3.8.0a0|>=3.6,<3.7.0a0']
      - imagesize -> python[version='2.7.*|3.5.*|3.6.*|3.4.*|>=3.5,<3.6.0a0|>=3.6,<3.7.0a0|>=3.7,<3.8.0a0|>=2.7,<2.8.0a0']
      - intervaltree -> python[version='2.7.*|3.5.*|3.6.*']
      - ipywidgets -> python[version='2.7.*|3.5.*|3.6.*|>=2.7,<2.8.0a0|>=3.5,<3.6.0a0|>=3.6,<3.7.0a0|3.4.*|>=3.7,<3.8.0a0']
      - itsdangerous -> python[version='2.7.*|3.5.*|3.6.*|3.4.*|>=3.6,<3.7.0a0|>=2.7,<2.8.0a0|>=3.7,<3.8.0a0|>=3.5,<3.6.0a0']
      - jdcal -> python[version='2.7.*|3.5.*|3.6.*|>=3.7,<3.8.0a0|>=3.5,<3.6.0a0|>=2.7,<2.8.0a0|>=3.6,<3.7.0a0']
      - jinja2 -> python[version='2.7.*|3.5.*|3.6.*|3.4.*|>=3.7,<3.8.0a0|>=3.6,<3.7.0a0|>=2.7,<2.8.0a0|>=3.5,<3.6.0a0']
      - matplotlib -> python[version='3.6.*|<3']
      - mayavi -> python[version='2.7.*|3.4.*|3.5.*|3.6.*']
      - mkl-service -> python[version='2.7.*|3.5.*|3.6.*|3.4.*']
      - mkl_fft -> python=3.4
      - more-itertools -> python=3.4
      - navigator-updater -> python[version='2.7.*|3.5.*|3.6.*|3.4.*']
      - numpydoc -> python[version='2.7.*|3.5.*|3.6.*|3.4.*|>=3.7,<3.8.0a0|>=3.5,<3.6.0a0|>=3.6,<3.7.0a0|>=2.7,<2.8.0a0']
      - olefile -> python[version='2.7.*|3.5.*|3.6.*|3.4.*|>=3.5,<3.6.0a0|>=3.6,<3.7.0a0|>=3.7,<3.8.0a0|>=2.7,<2.8.0a0']
      - packaging -> python[version='2.7.*|3.4.*|3.5.*|3.6.*|>=3.6,<3.7.0a0|>=2.7,<2.8.0a0|>=3.7,<3.8.0a0|>=3.5,<3.6.0a0']
      - parso -> python[version='>=2.7,<2.8.0a0|>=3.6,<3.7.0a0|>=3.7,<3.8.0a0|>=3.5,<3.6.0a0']
      - pathtools -> python[version='2.7.*|3.4.*|3.5.*|3.6.*']
      - prometheus_client -> python[version='2.7.*|3.5.*|3.6.*|3.4.*|>=3.7,<3.8.0a0|>=2.7,<2.8.0a0|>=3.6,<3.7.0a0|>=3.5,<3.6.0a0']
      - py -> python[version='2.7.*|3.5.*|3.6.*|3.4.*|>=3.7,<3.8.0a0|>=2.7,<2.8.0a0|>=3.6,<3.7.0a0|>=3.5,<3.6.0a0']
      - pyparsing -> python[version='2.7.*|3.5.*|3.6.*|3.4.*|>=3.6,<3.7.0a0|>=3.7,<3.8.0a0|>=2.7,<2.8.0a0|>=3.5,<3.6.0a0']
      - python-dateutil -> python[version='2.7.*|3.5.*|3.6.*|3.4.*|>=3.7,<3.8.0a0|>=3.6,<3.7.0a0|>=2.7,<2.8.0a0|>=3.5,<3.6.0a0']
      - pytz -> python[version='2.7.*|3.5.*|3.6.*|3.4.*|>=3.6,<3.7.0a0|>=2.7,<2.8.0a0|>=3.7,<3.8.0a0|>=3.5,<3.6.0a0']
      - qtawesome -> python[version='2.7.*|3.5.*|3.6.*|3.4.*|>=3.6,<3.7.0a0|>=2.7,<2.8.0a0|>=3.7,<3.8.0a0|>=3.5,<3.6.0a0']
      - qtconsole -> python[version='2.7.*|3.5.*|3.6.*|>=2.7,<2.8.0a0|>=3.6,<3.7.0a0|>=3.7,<3.8.0a0|>=3.5,<3.6.0a0|3.4.*']
      - qtpy -> python[version='2.7.*|3.5.*|3.6.*|>=2.7,<2.8.0a0|>=3.5,<3.6.0a0|>=3.6,<3.7.0a0|3.4.*|>=3.7,<3.8.0a0']
      - rope -> python[version='2.7.*|3.5.*|3.6.*|>=3.6,<3.7.0a0|>=2.7,<2.8.0a0|>=3.7,<3.8.0a0|>=3.5,<3.6.0a0']
      - snowballstemmer -> python[version='2.7.*|3.4.*|3.5.*|3.6.*|>=3.7,<3.8.0a0|>=3.6,<3.7.0a0|>=3.5,<3.6.0a0|>=2.7,<2.8.0a0']
      - sortedcollections -> python[version='2.7.*|3.5.*|3.6.*|>=2.7,<2.8.0a0|>=3.5,<3.6.0a0|>=3.6,<3.7.0a0|3.4.*|>=3.7,<3.8.0a0']
      - sphinxcontrib-websupport -> python[version='2.7.*|3.5.*|3.6.*|>=2.7,<2.8.0a0|>=3.5,<3.6.0a0|>=3.6,<3.7.0a0|>=3.7,<3.8.0a0']
      - tblib -> python[version='2.7.*|3.5.*|3.6.*|>=3.7,<3.8.0a0|>=3.6,<3.7.0a0|>=3.5,<3.6.0a0|>=2.7,<2.8.0a0']
      - testpath -> python[version='2.7.*|3.5.*|3.6.*|>=2.7,<2.8.0a0|>=3.6,<3.7.0a0|>=3.7,<3.8.0a0|>=3.5,<3.6.0a0']
      - toml -> python[version='>=2.7,<2.8.0a0|>=3.6,<3.7.0a0|>=3.7,<3.8.0a0|>=3.5,<3.6.0a0']
      - toolz -> python[version='2.7.*|3.5.*|3.6.*|3.4.*|>=3.7,<3.8.0a0|>=3.5,<3.6.0a0|>=3.6,<3.7.0a0|>=2.7,<2.8.0a0']
      - werkzeug -> python[version='2.7.*|3.5.*|3.6.*|3.4.*|>=3.7,<3.8.0a0|>=3.5,<3.6.0a0|>=2.7,<2.8.0a0|>=3.6,<3.7.0a0']
      - xlrd -> python[version='2.7.*|3.5.*|3.6.*|>=3.6,<3.7.0a0|>=3.7,<3.8.0a0|>=2.7,<2.8.0a0|>=3.5,<3.6.0a0']
      - xlsxwriter -> python[version='2.7.*|3.5.*|3.6.*|>=2.7,<2.8.0a0|>=3.7,<3.8.0a0|>=3.6,<3.7.0a0|>=3.5,<3.6.0a0']
      - xmltodict -> python[version='2.7.*|3.5.*|3.6.*|3.4.*|>=3.5,<3.6.0a0|>=3.6,<3.7.0a0|>=2.7,<2.8.0a0|>=3.7,<3.8.0a0']
      - yapf -> python[version='2.7.*|3.5.*|3.6.*|3.4.*|>=3.7,<3.8.0a0|>=2.7,<2.8.0a0|>=3.6,<3.7.0a0|>=3.5,<3.6.0a0']
      - zict -> python[version='2.7.*|3.5.*|3.6.*|3.4.*|>=3.7,<3.8.0a0|>=3.6,<3.7.0a0|>=2.7,<2.8.0a0|>=3.5,<3.6.0a0']
      - zope.event -> python[version='2.7.*|3.5.*|3.6.*|>=3.5,<3.6.0a0|3.4.*']

    Your python: python=3.8

    If python is on the left-most side of the chain, that's the version you've asked for.
    When python appears to the right, that indicates that the thing on the left is somehow
    not available for the python version you are constrained to. Note that conda will not
    change your python version to a different minor version unless you explicitly specify
    that.

    The following specifications were found to be incompatible with each other:

    Output in format: Requested package -> Available versions

    Package bzip2 conflicts for:
    _ipyw_jlab_nb_ext_conf -> python[version='>=3.5,<3.6.0a0'] -> bzip2[version='>=1.0.6,<2.0a0|>=1.0.8,<2.0a0']
    ipython_genutils -> python -> bzip2[version='>=1.0.6,<2.0a0|>=1.0.8,<2.0a0']
    pyzmq -> python[version='>=3.6,<3.7.0a0'] -> bzip2[version='>=1.0.6,<2.0a0|>=1.0.8,<2.0a0']
    openpyxl -> python[version='>=3.6'] -> bzip2[version='>=1.0.6,<2.0a0|>=1.0.8,<2.0a0']
    pathtools -> python -> bzip2[version='>=1.0.6,<2.0a0|>=1.0.8,<2.0a0']
    python-language-server -> python[version='>=3.7,<3.8.0a0'] -> bzip2[version='>=1.0.6,<2.0a0|>=1.0.8,<2.0a0']
    xlwt -> python -> bzip2[version='>=1.0.6,<2.0a0|>=1.0.8,<2.0a0']
    sqlalchemy -> pypy3.6[version='>=7.3.1'] -> bzip2[version='>=1.0.6,<2.0a0|>=1.0.8,<2.0a0']
    mkl_fft -> python[version='>=3.6,<3.7.0a0'] -> bzip2[version='>=1.0.6,<2.0a0|>=1.0.8,<2.0a0']
    idna -> python -> bzip2[version='>=1.0.6,<2.0a0|>=1.0.8,<2.0a0']
    jsonschema -> python[version='>=3.6,<3.7.0a0'] -> bzip2[version='>=1.0.6,<2.0a0|>=1.0.8,<2.0a0']





Try with miniconda and py37 instead of anaconda3 and py38
-----------------------------------------------------------

* https://repo.anaconda.com/miniconda/
* https://repo.anaconda.com/miniconda/Miniconda3-py37_4.8.3-MacOSX-x86_64.sh









EOU
}
condaforge-get(){
   local dir=$(dirname $(condaforge-dir)) &&  mkdir -p $dir && cd $dir

}
