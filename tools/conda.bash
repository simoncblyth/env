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

conda 
-----------------

* https://conda.io/docs/index.html
* https://conda.io/docs/user-guide/index.html
* https://conda.io/docs/commands/conda-search.html


::

    epsilon:~ blyth$ conda info --envs
    # conda environments:
    #
    base                  *  /usr/local/env/tools/conda/miniconda3

    epsilon:~ blyth$ 

    epsilon:~ blyth$ conda create --name snowflakes
    Solving environment: done

    ## Package Plan ##

      environment location: /usr/local/env/tools/conda/miniconda3/envs/snowflakes


    Proceed ([y]/n)? 

    Preparing transaction: done
    Verifying transaction: done
    Executing transaction: done
    #
    # To activate this environment, use:
    # > source activate snowflakes
    #
    # To deactivate an active environment, use:
    # > source deactivate
    #

    epsilon:~ blyth$ source activate snowflakes
    (snowflakes) epsilon:~ blyth$ source deactivate
    epsilon:~ blyth$

    epsilon:~ blyth$ source activate
    (base) epsilon:~ blyth$ source deactivate
    epsilon:~ blyth$ 




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

* for full logs see ~/conda/

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
    ...
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



conda basics
--------------

::

    conda install scipy
    conda install ipython
    conda install sympy 
    conda install matplotlib


conda install of anything taking 30 min at a minimum
-----------------------------------------------------------

* https://github.com/conda/conda/issues/7690




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
