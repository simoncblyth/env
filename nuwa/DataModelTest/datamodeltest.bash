# === func-gen- : nuwa/DataModelTest/datamodeltest fgp nuwa/DataModelTest/datamodeltest.bash fgn datamodeltest fgh nuwa/DataModelTest
datamodeltest-src(){      echo nuwa/DataModelTest/datamodeltest.bash ; }
datamodeltest-source(){   echo ${BASH_SOURCE:-$(env-home)/$(datamodeltest-src)} ; }
datamodeltest-vi(){       vi $(datamodeltest-source) ; }
datamodeltest-env(){      
   elocal- 
   rootsys-
}
datamodeltest-usage(){ cat << EOU

DataModelTest
================== 

Testing external usage of subset of NuWa DataModel, usage::

    datamodeltest-;datamodeltest-build-full

    delta:~ blyth$ which DataModelTest
    /usr/local/env/nuwa/bin/DataModelTest

    delta:~ blyth$ DataModelTest
    DayaBayAD1
    DayaBayAD2
    DayaBayAD3
    ...


RPATH setup/debug
-------------------

Initially cannot find libs without envvars::

    (chroma_env)delta:DataModelTest blyth$ /usr/local/env/bin/DataModelTest
    dyld: Library not loaded: libDataModel.dylib
      Referenced from: /usr/local/env/bin/DataModelTest
      Reason: image not found
    Trace/BPT trap: 5

After enable RPATH in the lib and exe, runtime knows where to look for libs::

    (chroma_env)delta:~ blyth$ otool-;otool-rpath /usr/local/env/nuwa/bin/DataModelTest | grep path
             path /usr/local/env/chroma_env/src/root-v5.34.14/lib (offset 12)
             path /usr/local/env/nuwa/lib (offset 12)

Look no hands::

    delta:~ blyth$ DYLD_LIBRARY_PATH= LD_LIBRARY_PATH= DataModelTest
    DayaBayAD1
    DayaBayAD2
    ...

Binary can be moved around::

    delta:~ blyth$ cp $(which DataModelTest) /tmp/
    delta:~ blyth$ /tmp/DataModelTest
    DayaBayAD1
    DayaBayAD2
    ...

BUT suspect the current cmake RPATH config restricts 
where the initial install can be made, eg could 
not install it to /tmp and get it to run ?



EOU
}
datamodeltest-dir(){ echo $(env-home)/nuwa/DataModelTest ; }
datamodeltest-tmpdir(){ echo /tmp/env/nuwa/DataModelTest ; }
datamodeltest-prefix(){  echo $LOCAL_BASE/env/nuwa ; }
#datamodeltest-prefix(){  echo $LOCAL_BASE/env ; }   # RPATH loading fails when installed here
datamodeltest-cd(){  cd $(datamodeltest-dir); }
datamodeltest-tcd(){  cd $(datamodeltest-tmpdir); }

####### manual build

datamodeltest-compile(){
   datamodeltest-cd

   local clhep=$(dirname $(chroma-clhep-incdir))   # G4 incdir,risky 
   local name=test

   clang -c \
         -I$(datamodel-prefix)/include \
         -I$(chroma-root-incdir) \
         -I$clhep \
         -DGOD_NOALLOC \
         $name.cc


   clang  -L$(datamodel-prefix)/lib -lDataModel \
          /usr/local/env/chroma_env/lib/libG4clhep.dylib \
         $name.o
}



###### cmake build

datamodeltest-cmake(){
    local iwd=$PWD
    mkdir -p $(datamodeltest-tmpdir)
    datamodeltest-tcd
    cmake -DCMAKE_INSTALL_PREFIX=$(datamodeltest-prefix) $(datamodeltest-dir)
    cd $iwd
}

datamodeltest-make(){
    local iwd=$PWD
    datamodeltest-tcd
    make $*
    cd $iwd
}
datamodeltest-install(){
    datamodeltest-make install
}
datamodeltest-build(){
    datamodeltest-cmake
    datamodeltest-make
    datamodeltest-install
}
datamodeltest-wipe(){
    rm -rf $(datamodeltest-tmpdir)
}
datamodeltest-build-full(){
    datamodeltest-wipe
    datamodeltest-build
}

