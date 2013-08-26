# === func-gen- : base/gcc fgp base/gcc.bash fgn gcc fgh base
gcc-src(){      echo base/gcc.bash ; }
gcc-source(){   echo ${BASH_SOURCE:-$(env-home)/$(gcc-src)} ; }
gcc-vi(){       vi $(gcc-source) ; }
gcc-env(){      elocal- ; }
gcc-usage(){ cat << EOU


MULTIPLE gcc/g++ HANDLING
============================

* http://gcc.gnu.org/
* http://gcc.gnu.org/releases.html



update-alternatives
---------------------

* http://linux.die.net/man/8/update-alternatives

::

    [blyth@belle7 dyb]$ update-alternatives --help
    alternatives version 1.3.30.1 - Copyright (C) 2001 Red Hat, Inc.
    This may be freely redistributed under the terms of the GNU Public License.

    usage: alternatives --install <link> <name> <path> <priority>
                        [--initscript <service>]
                        [--slave <link> <name> <path>]*
           alternatives --remove <name> <path>
           alternatives --auto <name>
           alternatives --config <name>
           alternatives --display <name>
           alternatives --set <name> <path>

    common options: --verbose --test --help --usage --version
                    --altdir <directory> --admindir <directory>


configure 4.8.1, pre-requisites
----------------------------------

::

    checking for the correct version of gmp.h... no
    configure: error: Building GCC requires GMP 4.2+, MPFR 2.4.0+ and MPC 0.8.0+.
    Try the --with-gmp, --with-mpfr and/or --with-mpc options to specify
    their locations.  Source code for these libraries can be found at
    their respective hosting sites as well as at
    ftp://gcc.gnu.org/pub/gcc/infrastructure/.  See also
    http://gcc.gnu.org/install/prerequisites.html for additional info.  If
    you obtained GMP, MPFR and/or MPC from a vendor distribution package,
    make sure that you have installed both the libraries and the header
    files.  They may be located in separate packages.


4.8.1 includes ./contrib/download_prerequisites which grabs
GMP, MPFR and MPC. After this the configure completes.



EOU
}
gcc-dir(){ echo $(local-base)/env/base/$(gcc-name) ; }
gcc-cd(){  cd $(gcc-dir); }
gcc-mate(){ mate $(gcc-dir) ; }
gcc-name(){  echo gcc-4.8.1 ; }
#gcc-name(){  echo gcc-4.5.4 ; }
gcc-mirror(){ echo ${GCC_MIRROR:- http://ftp.tsukuba.wide.ad.jp/software/gcc/releases} ;}
gcc-url(){ echo $(gcc-mirror)/$1/$1.tar.gz ; }
gcc-get(){
   local dir=$(dirname $(gcc-dir)) &&  mkdir -p $dir && cd $dir
   local nam=$(gcc-name)
   local tgz=$nam.tar.gz
   [ ! -f "$tgz" ] && curl -L -O $(gcc-url $nam)
   [ ! -d "$nam" ] && tar zxvf $tgz
}

gcc-prefix(){ echo /usr/local/env ; }

gcc-configure(){
  gcc-cd
  local pfx=$(gcc-prefix)
  [ ! -d "$pfx" ] && echo $msg ERROR prefix dir $pfx does not exist && return 1 
  ./configure --prefix=$pfx
}

gcc-make(){
  gcc-cd
  make 
}

