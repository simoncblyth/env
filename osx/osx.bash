# === func-gen- : osx/osx fgp osx/osx.bash fgn osx fgh osx
osx-src(){      echo osx/osx.bash ; }
osx-source(){   echo ${BASH_SOURCE:-$(env-home)/$(osx-src)} ; }
osx-vi(){       vi $(osx-source) ; }
osx-env(){      elocal- ; }
osx-usage(){ cat << EOU




FUNCTIONS
-----------

osx-library-visible
       http://gregferro.com/make-library-folder-visible-in-os-x-lion/
       http://coolestguidesontheplanet.com/show-hidden-library-and-user-library-folder-in-osx/


EOU
}
osx-dir(){ echo $(local-base)/env/osx/osx-osx ; }
osx-cd(){  cd $(osx-dir); }
osx-mate(){ mate $(osx-dir) ; }
osx-get(){
   local dir=$(dirname $(osx-dir)) &&  mkdir -p $dir && cd $dir

}


osx-library-visible(){
 
   chflags nohidden ~/Library/
}


