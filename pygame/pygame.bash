# === func-gen- : pygame/pygame fgp pygame/pygame.bash fgn pygame fgh pygame
pygame-src(){      echo pygame/pygame.bash ; }
pygame-source(){   echo ${BASH_SOURCE:-$(env-home)/$(pygame-src)} ; }
pygame-vi(){       vi $(pygame-source) ; }
pygame-env(){      elocal- ; }
pygame-usage(){ cat << EOU

PYGAME
=======






EOU
}
pygame-dir(){ echo $(local-base)/env/pygame/pygame-pygame ; }
pygame-cd(){  cd $(pygame-dir); }
pygame-mate(){ mate $(pygame-dir) ; }
pygame-get(){
   local dir=$(dirname $(pygame-dir)) &&  mkdir -p $dir && cd $dir

}

pygame-docs(){ open /opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/pygame/docs/index.html ; }

