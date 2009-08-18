# === func-gen- : dj/djextensions fgp dj/djextensions.bash fgn djextensions fgh dj
djextensions-src(){      echo dj/djextensions.bash ; }
djextensions-source(){   echo ${BASH_SOURCE:-$(env-home)/$(djextensions-src)} ; }
djextensions-vi(){       vi $(djextensions-source) ; }
djextensions-env(){      elocal- ; }
djextensions-usage(){
  cat << EOU
     djextensions-src : $(djextensions-src)
     djextensions-dir : $(djextensions-dir)

    Include django_extensions in INSTALLED_APPS , then list available commands with :
      ./manage.py help

    Particularly handy ones :

      ./manage.py shell_plus       ## import models from all installed apps 
  

EOU
}
djextensions-dir(){ echo $(local-base)/env/django/django-command-extensions ; }
djextensions-cd(){  cd $(djextensions-dir); }
djextensions-mate(){ mate $(djextensions-dir) ; }
djextensions-get(){
   local dir=$(dirname $(djextensions-dir)) &&  mkdir -p $dir && cd $dir
   hg clone http://hgsvn.trbs.net/django-command-extensions
}

djextensions-ln(){
   python-ln $(djextensions-dir)/django_extensions
}

