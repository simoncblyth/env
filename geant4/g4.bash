# === func-gen- : geant4/g4 fgp geant4/g4.bash fgn g4 fgh geant4
g4-src(){      echo geant4/g4.bash ; }
g4-source(){   echo ${BASH_SOURCE:-$(env-home)/$(g4-src)} ; }
g4-vi(){       vi $(g4-source) ; }
g4-env(){      elocal- ; nuwa- ;  }
g4-usage(){ cat << EOU

#. Former g4-vrml- moved to vrml-


EOU
}
g4-dir(){ echo $(nuwa-g4-bdir); }
g4-cd(){  cd $(g4-dir)/$1; }
g4-mate(){ mate $(g4-dir) ; }
g4-get(){
   local dir=$(dirname $(g4-dir)) &&  mkdir -p $dir && cd $dir

}

g4-gdml(){ g4-cd source/persistency/gdml/src/$1 ; }



g4-rebuild-env(){
   fenv
   cd $(nuwa-g4-cmtdir)
   cmt config
   . setup.sh
}


g4-libs-marker(){ echo $G4INSTALL/lib/$G4SYSTEM/libG4run.so ; }
g4-libs-ls(){ ls -l $G4INSTALL/lib/$G4SYSTEM ; }
g4-libs-rebuild(){

   g4-rebuild-env

   local marker=$(g4-libs-marker)
   if [ -f "$marker" ]; then 
      echo $msg : removing $marker
      rm -rf $marker
   fi   
   cmt pkg_make 

}


g4-libname-marker(){ echo $G4INSTALL/lib/$G4SYSTEM/libname.map ; }
g4-libname-ls(){  ls -l $(g4-libname-marker) ; }
g4-libname-rebuild(){

   g4-rebuild-env

   local marker=$(g4-libname-marker)
   if [ -f "$marker" ]; then 
      echo $msg : removing $marker
      rm -rf $marker
   fi   
   cmt pkg_make 

}



g4-includes-marker(){ echo $G4INSTALL/include/G4Version.hh ; }
g4-includes-ls(){  ls -l $G4INSTALL/include ; }
g4-includes-rebuild(){

   g4-rebuild-env

   local marker=$(g4-includes-marker)
   if [ -f "$marker" ]; then 
      echo $msg : removing $marker
      rm -rf $marker
   fi   
   cmt pkg_make 

}

