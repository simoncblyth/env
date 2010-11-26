# === func-gen- : gui/gnocl fgp gui/gnocl.bash fgn gnocl fgh gui
gnocl-src(){      echo gui/gnocl.bash ; }
gnocl-source(){   echo ${BASH_SOURCE:-$(env-home)/$(gnocl-src)} ; }
gnocl-vi(){       vi $(gnocl-source) ; }
gnocl-env(){      elocal- ; }
gnocl-usage(){
  cat << EOU
     gnocl-src : $(gnocl-src)
     gnocl-dir : $(gnocl-dir)




    http://www.dr-baum.net/gnocl/index.html#what 

EOU
}
gnocl-dir(){ echo $(local-base)/env/gui/gui-gnocl ; }
gnocl-cd(){  cd $(gnocl-dir); }
gnocl-mate(){ mate $(gnocl-dir) ; }
gnocl-get(){
   local dir=$(dirname $(gnocl-dir)) &&  mkdir -p $dir && cd $dir

}
