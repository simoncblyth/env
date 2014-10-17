# === func-gen- : nuwa/DataModelTest/datamodeltest fgp nuwa/DataModelTest/datamodeltest.bash fgn datamodeltest fgh nuwa/DataModelTest
datamodeltest-src(){      echo nuwa/DataModelTest/datamodeltest.bash ; }
datamodeltest-source(){   echo ${BASH_SOURCE:-$(env-home)/$(datamodeltest-src)} ; }
datamodeltest-vi(){       vi $(datamodeltest-source) ; }
datamodeltest-env(){      
   elocal- 
   datamodel-
   chroma-
   chroma-geant4-export   # for GEANT4_HOME used to access FindROOT, TODO: clean this up
}
datamodeltest-usage(){ cat << EOU


rpath not working::

    (chroma_env)delta:DataModelTest blyth$ /usr/local/env/bin/DataModelTest
    dyld: Library not loaded: libDataModel.dylib
      Referenced from: /usr/local/env/bin/DataModelTest
      Reason: image not found
    Trace/BPT trap: 5


    (chroma_env)delta:build blyth$ DYLD_LIBRARY_PATH=/usr/local/env/nuwa/lib /usr/local/env/bin/DataModelTest
    DayaBayAD1
    DayaBayAD2
    DayaBayAD3
    DayaBayAD4
    DayaBayIWS
    DayaBayOWS
    LingAoAD1
    LingAoAD2
    LingAoAD3
    LingAoAD4
    LingAoIWS
    LingAoOWS
    FarAD1
    FarAD2
    FarAD3
    FarAD4
    FarIWS
    FarOWS




EOU
}
datamodeltest-dir(){ echo $(env-home)/nuwa/DataModelTest ; }
datamodeltest-tmpdir(){ echo /tmp/env/nuwa/DataModelTest ; }
datamodeltest-prefix(){  echo $LOCAL_BASE/env ; }
datamodeltest-cd(){  cd $(datamodeltest-dir); }
datamodeltest-tcd(){  cd $(datamodeltest-tmpdir); }

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


datamodeltest-cmake(){
    mkdir -p $(datamodeltest-tmpdir)
    datamodeltest-tcd
    cmake -DCMAKE_INSTALL_PREFIX=$(datamodeltest-prefix) $(datamodeltest-dir)
}

datamodeltest-make(){
    datamodeltest-tcd
    make $*
}
datamodeltest-install(){
    datamodeltest-make install
}
datamodeltest-build(){
    datamodeltest-cmake
    datamodeltest-make
    datamodeltest-install
}


