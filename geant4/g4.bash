# === func-gen- : geant4/g4 fgp geant4/g4.bash fgn g4 fgh geant4
g4-src(){      echo geant4/g4.bash ; }
g4-source(){   echo ${BASH_SOURCE:-$(env-home)/$(g4-src)} ; }
g4-vi(){       vi $(g4-source) ; }
g4-env(){      elocal- ; nuwa- ;  }
g4-usage(){ cat << EOU

#. Former g4-vrml- moved to vrml-
#. see related notes :doc:`geant4/geant4_patch`

g4-install-rebuild

    requires manual step to take action, 
    as this is kinda expensive to recover from if done by mistake::

        [blyth@belle7 4.9.2.p01]$ pwd
        /data1/env/local/dyb/external/geant4/4.9.2.p01
        [blyth@belle7 4.9.2.p01]$ mv i686-slc5-gcc41-dbg i686-slc5-gcc41-dbg.prior 



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



g4-install-rebuild(){

   g4-rebuild-env

   echo NEEDS A MANUAL STEP TO INSTALL NEW GEANT4 LIBS AND INCLUDES INTO LCG_DESTDIR

   cmt pkg_install

}


g4-install-ls(){
   g4-rebuild-env
   echo -----------
   ls -l ${LCG_destdir}/lib
   echo -----------
   echo LCG_destdir ${LCG_destdir} 
   echo LCG_destdir/lib ${LCG_destdir}/lib  COUNT SO LIBS : $(ls -1 ${LCG_destdir}/lib/lib*.so | wc -l )
   #echo LCG_destdir/include ${LCG_destdir}/include  COUNT HH INCS : $(ls -1 ${LCG_destdir}/include/*.hh | wc -l )
}


