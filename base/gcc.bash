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



debugging options
-------------------

Using L7 /usr/bin/gdb with gcc1120 code gives::

    Reading symbols from /hpcfs/juno/junogpu/blyth/local/Opticks-0.0.1_alpha/x86_64-CentOS7-gcc1120-geant4_10_04_p02-dbg/lib/SArTest...
    Dwarf Error: wrong version in compilation unit header (is 5, should be 2, 3, or 4) 
    [in module /hpcfs/juno/junogpu/blyth/local/Opticks-0.0.1_alpha/x86_64-CentOS7-gcc1120-geant4_10_04_p02-dbg/lib/SArTest]
    (no debugging symbols found)...done.
    (gdb) r

Incompatibility between::

    /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/contrib/gcc/11.2.0/bin/gcc
    /usr/bin/gdb # on L7

* http://gcc.gnu.org/onlinedocs/gcc/Debugging-Options.html


-gdwarf-version

    Produce debugging information in DWARF format (if that is supported). The
    value of version may be either 2, 3, 4 or 5; the default version for most
    targets is 5 (with the exception of VxWorks, TPF and Darwin / macOS, which
    default to version 2, and AIX, which defaults to version 4).

    Note that with DWARF Version 2, some ports require and always use some
    non-conflicting DWARF 3 extensions in the unwind tables.

    Version 4 may require GDB 7.0 and -fvar-tracking-assignments for maximum benefit. 
    Version 5 requires GDB 8.0 or higher.

    GCC no longer supports DWARF Version 1, which is substantially different than Version 2 and later. For historical reasons, some other DWARF-related options such as -fno-dwarf2-cfi-asm) retain a reference to DWARF Version 2 in their names, but apply to all currently-supported versions of DWARF.

::

    L7[blyth@lxslc712 mandelbrot]$ as --help | grep dwarf
      --gdwarf-<N>            generate DWARF<N> debugging information. 2 <= <N> <= 5
      --gdwarf-sections       generate per-function section names for DWARF line information
    L7[blyth@lxslc712 mandelbrot]$ as --help | grep stabs
      --gstabs                generate STABS debugging information
      --gstabs+               generate STABS debug info with GNU extensions
    L7[blyth@lxslc712 mandelbrot]$ 



* https://gcc.gnu.org/gcc-11/changes.html

::

   To make GCC 11 generate an older DWARF version use -g together with -gdwarf-2, -gdwarf-3 or -gdwarf-4. 



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

