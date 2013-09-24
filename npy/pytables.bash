# === func-gen- : npy/pytables fgp npy/pytables.bash fgn pytables fgh npy
pytables-src(){      echo npy/pytables.bash ; }
pytables-source(){   echo ${BASH_SOURCE:-$(env-home)/$(pytables-src)} ; }
pytables-vi(){       vi $(pytables-source) ; }
pytables-env(){      elocal- ; }
pytables-usage(){ cat << EOU

PYTABLES
==========

Pre-requisites
----------------

#. HDF5, at least HDF5 1.8.4,  http://www.hdfgroup.org/HDF5/
#. NumPy, at least NumPy 1.4.1, http://www.numpy.org 
#. Numexpr, at least Numexpr 2.9, http://code.google.com/p/numexpr

Optional
---------

#. LZO compression, http://www.oberhumer.com/opensource/
#. bzip2, http://www.bzip.org/ 


::

    simon:e blyth$ port info hdf5-18  
    hdf5-18 @1.8.11 (science)
    Variants:             [+]cxx, fortran, gcc44, gcc45, gcc46, gcc47, gcc48, mpich, mpich2, openmpi, szip, threadsafe, universal

    Description:          HDF5 is a data model, library, and file format for storing and managing data. It supports an unlimited variety of datatypes, and is designed for flexible and efficient I/O and for high volume and complex data. HDF5 is portable and is extensible,
                          allowing applications to evolve in their use of HDF5. The HDF5 Technology suite includes tools and applications for managing, manipulating, viewing, and analyzing data in the HDF5 format.
    Homepage:             http://www.hdfgroup.org/HDF5/

    Library Dependencies: zlib
    Conflicts with:       hdf5
    Platforms:            darwin
    License:              NCSA
    Maintainers:          mmoll@macports.org, openmaintainer@macports.org




EOU
}
pytables-name(){ echo tables-3.0.0 ; }
pytables-url(){  echo http://downloads.sourceforge.net/project/pytables/pytables/3.0.0/tables-3.0.0.tar.gz ; }
pytables-dir(){ echo $(local-base)/env/npy/$(pytables-name) ; }
pytables-cd(){  cd $(pytables-dir); }
pytables-mate(){ mate $(pytables-dir) ; }
pytables-get(){
   local dir=$(dirname $(pytables-dir)) &&  mkdir -p $dir && cd $dir

   local url=$(pytables-url)
   local nam=$(pytables-name)
   local tgz=$(basename $url)
   [ ! -f "$tgz" ] && curl -L -O $url
   [ ! -d "$nam" ] && tar zxvf $tgz

}
