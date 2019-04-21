# === func-gen- : tools/CLI11/cli fgp tools/CLI11/cli.bash fgn cli fgh tools/CLI11 src base/func.bash
cli11-source(){   echo ${BASH_SOURCE} ; }
cli11-edir(){ echo $(dirname $(cli11-source)) ; }
cli11-ecd(){  cd $(cli11-edir); }
cli11-dir(){  echo $LOCAL_BASE/env/tools/cli/cli11 ; }
cli11-cd(){   cd $(cli11-dir); }
cli11-vi(){   vi $(cli11-source) ; }
cli11-env(){  elocal- ; }
cli11-usage(){ cat << EOU

CLI11
=======


* this is feeling a bit heavy for a commandline parser




Tring to build just tests, given that its header only. But no gives "Invalid escape sequence \."
and missing cmake required version.

::

    208 add_subdirectory(subcom_in_files)
    209 add_test(NAME subcom_in_files COMMAND subcommand_main subcommand_a -f this.txt --with-foo)
    210 set_property(TEST subcom_in_files PROPERTY PASS_REGULAR_EXPRESSION
    211     "Working on file: this\.txt"
    212     "Using foo!")



So build from top
------------------

::

    [blyth@localhost cli]$ mkdir build
    [blyth@localhost cli]$ cd build
    [blyth@localhost build]$ cmake ..
    -- The CXX compiler identification is GNU 4.8.5
    -- Check for working CXX compiler: /usr/bin/c++
    -- Check for working CXX compiler: /usr/bin/c++ -- works
    -- Detecting CXX compiler ABI info
    -- Detecting CXX compiler ABI info - done
    -- Detecting CXX compile features
    -- Detecting CXX compile features - done
    CMake Error at tests/CMakeLists.txt:2 (message):
      You have requested tests be built, but googletest is not downloaded.
      Please run:

          git submodule update --init


    -- Configuring incomplete, errors occurred!
    See also "/home/blyth/local/env/tools/CLI11/cli/build/CMakeFiles/CMakeOutput.log".
    [blyth@localhost build]$ 



Externals : its using same nlohmann/json as Opticks NPY gets from YoctoGL
---------------------------------------------------------------------------------

::

    [blyth@localhost cli]$ git submodule update --init
    Submodule 'extern/googletest' (git@github.com:google/googletest.git) registered for path 'extern/googletest'
    Submodule 'extern/json' (git@github.com:nlohmann/json.git) registered for path 'extern/json'
    Submodule 'extern/sanitizers' (git@github.com:arsenm/sanitizers-cmake) registered for path 'extern/sanitizers'
    Cloning into 'extern/googletest'...
    remote: Enumerating objects: 16724, done.
    remote: Total 16724 (delta 0), reused 0 (delta 0), pack-reused 16724
    Receiving objects: 100% (16724/16724), 5.76 MiB | 2.96 MiB/s, done.
    Resolving deltas: 100% (12335/12335), done.
    Submodule path 'extern/googletest': checked out 'ec44c6c1675c25b9827aacd08c02433cccde7780'
    Cloning into 'extern/json'...
    remote: Enumerating objects: 46998, done.
    remote: Total 46998 (delta 0), reused 0 (delta 0), pack-reused 46998
    Receiving objects: 100% (46998/46998), 173.54 MiB | 6.09 MiB/s, done.
    Resolving deltas: 100% (38062/38062), done.
    Submodule path 'extern/json': checked out 'db53bdac1926d1baebcb459b685dcd2e4608c355'
    Cloning into 'extern/sanitizers'...
    Warning: Permanently added the RSA host key for IP address '13.250.177.223' to the list of known hosts.
    remote: Enumerating objects: 214, done.
    remote: Total 214 (delta 0), reused 0 (delta 0), pack-reused 214
    Receiving objects: 100% (214/214), 47.80 KiB | 0 bytes/s, done.
    Resolving deltas: 100% (136/136), done.
    Submodule path 'extern/sanitizers': checked out '6947cff3a9c9305eb9c16135dd81da3feb4bf87f'
    [blyth@localhost cli]$ 




EOU
}
cli11-get(){
   local dir=$(dirname $(cli11-dir)) &&  mkdir -p $dir && cd $dir

   [ ! -d cli11 ] && git clone git@github.com:simoncblyth/CLI11.git cli11

}

cli11-externals()
{
   cli11-cd
   git submodule update --init
}

cli11-bdir(){ echo $(cli11-dir).build ; }
cli11-bcd(){  cd $(cli11-bdir) ; }

cli11-cmake()
{
   local iwd=$PWD
   local bdir=$(cli11-bdir)
   local sdir=$(cli11-dir)
   mkdir -p $bdir && cd $bdir 

   cmake $sdir

   cd $iwd
}

cli11-make()
{
   cli11-bcd
   make $*
}

cli11--()
{
   cli11-get
   cli11-externals
   cli11-cmake
   cli11-make
}


