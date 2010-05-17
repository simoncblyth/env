# === func-gen- : trac/bitn fgp trac/bitn.bash fgn bitn fgh trac
bitn-src(){      echo trac/bitn.bash ; }
bitn-source(){   echo ${BASH_SOURCE:-$(env-home)/$(bitn-src)} ; }
bitn-vi(){       vi $(bitn-source) ; }
bitn-env(){      elocal- ; }
bitn-usage(){
  cat << EOU
     bitn-src : $(bitn-src)
     bitn-dir : $(bitn-dir)

   TAKE A LOOK AT THE LATEST BITTEN-SLAVE 

EOU
}
bitn-dir(){ echo $(local-base)/env/trac/bitn ; }
bitn-cd(){  cd $(bitn-dir); }
bitn-mate(){ mate $(bitn-dir) ; }


bitn-url(){ echo http://svn.edgewall.org/repos/bitten/trunk/ ; }
bitn-get(){
   local dir=$(dirname $(bitn-dir)) &&  mkdir -p $dir && cd $dir
   svn co $(bitn-url) bitn
}
