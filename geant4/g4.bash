# === func-gen- : geant4/g4 fgp geant4/g4.bash fgn g4 fgh geant4
g4-src(){      echo geant4/g4.bash ; }
g4-source(){   echo ${BASH_SOURCE:-$(env-home)/$(g4-src)} ; }
g4-vi(){       vi $(g4-source) ; }
g4-env(){      elocal- ; nuwa- ;  }
g4-usage(){ cat << EOU





EOU
}
g4-dir(){ echo $(nuwa-g4-bdir); }
g4-cd(){  cd $(g4-dir)/$1; }
g4-mate(){ mate $(g4-dir) ; }
g4-get(){
   local dir=$(dirname $(g4-dir)) &&  mkdir -p $dir && cd $dir

}

g4-gdml(){ g4-cd source/persistency/gdml/src/$1 ; }


g4-vrml(){ g4-cd source/visualization/VRML/src/$1 ; }
g4-vrml-dir(){ echo $(g4-dir)/source/visualization/VRML ; }
g4-vrml-libname(){ echo libG4VRML.so ; }

g4-vrml-deploy(){
  cd $(env-home)/geant4/source
  DEPLOY_MODE="diff" ./deploy.sh 
  DEPLOY_MODE="cp" ./deploy.sh 
}

g4-vrml-ls(){
  cd $(g4-vrml-dir)
  local name=$(g4-vrml-libname)
  local lib=../../../lib/Linux-g++/$name
  ls -l $lib $(nuwa-g4-libdir)/$name
}

g4-vrml-make(){
  local msg="=== $FUNCNAME :" 
  type $FUNCNAME

  cd $(g4-vrml-dir)

  local name=$(g4-vrml-libname)
  local lib=../../../lib/Linux-g++/$name
  echo $msg before
  ls -l $lib $(nuwa-g4-libdir)/$name

  export VERBOSE=1
  rm -f $lib && make CLHEP_BASE_DIR=$(nuwa-clhep-idir) G4SYSTEM=Linux-g++ G4LIB_BUILD_SHARED=1 && cp $lib $(nuwa-g4-libdir)/

  echo $msg after
  ls -l $lib $(nuwa-g4-libdir)/$name
}






