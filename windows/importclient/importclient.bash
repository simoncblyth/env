# === func-gen- : windows/importclient/importclient fgp windows/importclient/importclient.bash fgn importclient fgh windows/importclient
importclient-src(){      echo windows/importclient/importclient.bash ; }
importclient-source(){   echo ${BASH_SOURCE:-$(env-home)/$(importclient-src)} ; }
importclient-vi(){       vi $(importclient-source) ; }
importclient-usage(){ cat << EOU





EOU
}
importclient-sdir(){   echo $(env-home)/windows/importclient ; }
importclient-dir(){    echo $(local-base)/env/windows/importclient ; }
importclient-bdir(){   echo $(local-base)/env/windows/importclient/build ; }
importclient-prefix(){ echo $(local-base)/env/windows/importclient ; }

importclient-edit(){
   importclient-scd
   vi *
}

importclient-cd(){   cd $(importclient-dir); }
importclient-bcd(){  cd $(importclient-bdir); }
importclient-scd(){  cd $(importclient-sdir); }

importclient-name(){ echo DemoClient ; }
importclient-sln(){ echo $(importclient-bdir)/$(importclient-name).sln ; }
importclient-slnwin(){ echo $(vs-gitbash2win $(importclient-sln)) ; }
importclient-env(){      elocal- ; vs- ; }

importclient-cmake(){

   local bdir=$(importclient-bdir)
   rm -rf $bdir
   mkdir -p $bdir

   importlib-
   importclient-bcd
   cmake \
          -DCMAKE_INSTALL_PREFIX="$(importclient-prefix)" \
          -DDEMO_INCLUDE_DIRS="$(importlib-dir)" \
          -DDEMO_LIBRARIES=$(importlib-lib) \
          $(importclient-sdir)

}

importclient--(){
   importclient-bcd
   cmake --build . --config Debug --target ALL_BUILD
}


importclient-run(){

   importlib-
   PATH=$(importlib-libdir):"$PATH" $(importclient-bdir)/Debug/DemoClient.exe 
}



