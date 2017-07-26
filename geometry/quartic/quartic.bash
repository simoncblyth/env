# === func-gen- : geometry/quartic/quartic fgp geometry/quartic/quartic.bash fgn quartic fgh geometry/quartic
quartic-src(){      echo geometry/quartic/quartic.bash ; }
quartic-source(){   echo ${BASH_SOURCE:-$(env-home)/$(quartic-src)} ; }
quartic-vi(){       vi $(quartic-source) ; }
quartic-env(){      elocal- ; }
quartic-usage(){ cat << EOU





EOU
}
quartic-dir(){ echo $(env-home)/geometry/quartic/quartic ; }
quartic-c(){   cd $(quartic-dir); }
quartic-cd(){  cd $(quartic-dir); }
quartic-ecd(){  cd $(env-home)/geometry/quartic ; }
quartic-get(){
   local dir=$(dirname $(quartic-dir)) &&  mkdir -p $dir && cd $dir


   [ ! -f Roots3And4.c ] && curl -L -O http://www.realtimerendering.com/resources/GraphicsGems/gems/Roots3And4.c

    


}
