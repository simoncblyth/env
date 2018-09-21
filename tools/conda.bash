# === func-gen- : tools/conda fgp tools/conda.bash fgn conda fgh tools
conda-src(){      echo tools/conda.bash ; }
conda-source(){   echo ${BASH_SOURCE:-$(env-home)/$(conda-src)} ; }
conda-vi(){       vi $(conda-source) ; }
conda-env(){      elocal- ; }
conda-usage(){ cat << EOU

Conda 
========

Refs
-----

* http://jakevdp.github.io/blog/2016/08/25/conda-myths-and-misconceptions/
* https://github.com/conda-forge
* https://conda-forge.org/

Travis Oliphant (NumPy originator) on conda
----------------------------------------------- 

* http://technicaldiscovery.blogspot.tw/2013/12/why-i-promote-conda.html

Conda is an open-source, general, cross-platform package manager.  
One could accurately describe it as a
cross-platform hombrew written in Python.  Anyone can use the tool and
related infrastructure to build and distribute whatever packages they
want.

Linkers and Loaders : http://www.iecc.com/linker/



* https://conda.io/miniconda.html
* https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh


Install
---------

::

    /usr/local/env/tools/conda/miniconda3

      - Press ENTER to confirm the location
      - Press CTRL-C to abort the installation
      - Or specify a different location below

    [/usr/local/env/tools/conda/miniconda3] >>> 
    PREFIX=/usr/local/env/tools/conda/miniconda3
    installing: python-3.7.0-hc167b69_0 ...
    Python 3.7.0
    installing: ca-certificates-2018.03.07-0 ...
    installing: conda-env-2.6.0-1 ...
    installing: libcxxabi-4.0.1-hebd6815_0 ...
    installing: xz-5.2.4-h1de35cc_4 ...
    installing: yaml-0.1.7-hc338f04_2 ...
    installing: zlib-1.2.11-hf3cbc9b_2 ...
    installing: libcxx-4.0.1-h579ed51_0 ...
    installing: openssl-1.0.2p-h1de35cc_0 ...
    installing: tk-8.6.8-ha441bb4_0 ...
    installing: libffi-3.2.1-h475c297_4 ...
    installing: ncurses-6.1-h0a44026_0 ...
    installing: libedit-3.1.20170329-hb402a30_2 ...
    installing: readline-7.0-h1de35cc_5 ...
    installing: sqlite-3.24.0-ha441bb4_0 ...
    installing: asn1crypto-0.24.0-py37_0 ...
    installing: certifi-2018.8.24-py37_1 ...
    installing: chardet-3.0.4-py37_1 ...
    installing: idna-2.7-py37_0 ...
    installing: pycosat-0.6.3-py37h1de35cc_0 ...
    installing: pycparser-2.18-py37_1 ...
    installing: pysocks-1.6.8-py37_0 ...
    installing: python.app-2-py37_8 ...
    installing: ruamel_yaml-0.15.46-py37h1de35cc_0 ...
    installing: six-1.11.0-py37_1 ...
    installing: cffi-1.11.5-py37h6174b99_1 ...
    installing: setuptools-40.2.0-py37_0 ...
    installing: cryptography-2.3.1-py37hdbc3d79_0 ...
    installing: wheel-0.31.1-py37_0 ...
    installing: pip-10.0.1-py37_0 ...
    installing: pyopenssl-18.0.0-py37_0 ...
    installing: urllib3-1.23-py37_0 ...
    installing: requests-2.19.1-py37_0 ...
    installing: conda-4.5.11-py37_0 ...
    installation finished.
    WARNING:
        You currently have a PYTHONPATH environment variable set. This may cause
        unexpected behavior when running the Python interpreter in Miniconda3.
        For best results, please verify that your PYTHONPATH only points to
        directories of packages that are compatible with the Python interpreter
        in Miniconda3: /usr/local/env/tools/conda/miniconda3
    Do you wish the installer to prepend the Miniconda3 install location
    to PATH in your /Users/blyth/.bash_profile ? [yes|no]
    [yes] >>> yes

    Appending source /usr/local/env/tools/conda/miniconda3/bin/activate to /Users/blyth/.bash_profile
    A backup will be made to: /Users/blyth/.bash_profile-miniconda3.bak


    For this change to become active, you have to open a new terminal.

    Thank you for installing Miniconda3!
    epsilon:conda blyth$ 

::

    438 # added by Miniconda3 installer
    439 export PATH="/usr/local/env/tools/conda/miniconda3/bin:$PATH"



EOU
}
conda-dir(){ echo $(local-base)/env/tools/conda/miniconda3 ; }
conda-cd(){  cd $(conda-dir); }
conda-get(){
   local dir=$(dirname $(conda-dir)) &&  mkdir -p $dir && cd $dir

   local url=https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
   local nam=$(basename $url)

   [ ! -f "$nam" ] && curl -L -O $url

   [ ! -d miniconda3 ] && bash $nam -p $(conda-dir)


}
