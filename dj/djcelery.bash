# === func-gen- : dj/djcelery fgp dj/djcelery.bash fgn djcelery fgh dj
djcelery-src(){      echo dj/djcelery.bash ; }
djcelery-source(){   echo ${BASH_SOURCE:-$(env-home)/$(djcelery-src)} ; }
djcelery-vi(){       vi $(djcelery-source) ; }
djcelery-env(){      elocal- ; }
djcelery-usage(){
  cat << EOU
     djcelery-src : $(djcelery-src)
     djcelery-dir : $(djcelery-dir)


EOU
}
djcelery-dir(){ echo $(local-base)/env/dj/django-celery ; }
djcelery-cd(){  cd $(djcelery-dir); }
djcelery-mate(){ mate $(djcelery-dir) ; }
djcelery-get(){
   local dir=$(dirname $(djcelery-dir)) &&  mkdir -p $dir && cd $dir


   git clone git://github.com/ask/django-celery.git

}
