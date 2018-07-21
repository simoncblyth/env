# === func-gen- : video/imovie/imovie fgp video/imovie/imovie.bash fgn imovie fgh video/imovie
imovie-src(){      echo video/imovie/imovie.bash ; }
imovie-source(){   echo ${BASH_SOURCE:-$(env-home)/$(imovie-src)} ; }
imovie-vi(){       vi $(imovie-source) ; }
imovie-env(){      elocal- ; }
imovie-usage(){ cat << EOU

iMovie
========

issue : crashing on import of H.264 1920x1080 AAC 60fps
--------------------------------------------------------



EOU
}
imovie-dir(){ echo $(local-base)/env/video/imovie/video/imovie-imovie ; }
imovie-cd(){  cd $(imovie-dir); }
imovie-mate(){ mate $(imovie-dir) ; }
imovie-get(){
   local dir=$(dirname $(imovie-dir)) &&  mkdir -p $dir && cd $dir

}
