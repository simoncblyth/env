# === func-gen- : dj/djextensions fgp dj/djextensions.bash fgn djextensions fgh dj
djext-src(){      echo dj/djext.bash ; }
djext-source(){   echo ${BASH_SOURCE:-$(env-home)/$(djext-src)} ; }
djext-vi(){       vi $(djext-source) ; }
djext-env(){      elocal- ; }
djext-usage(){
  cat << EOU
     djext-src : $(djext-src)
     djext-dir : $(djext-dir)

    Pre-requisite : mercurial for hg , see hg-

    Install with 
         djext-get
         djext-ln
  

    Include django_extensions in INSTALLED_APPS , then list available commands with :
      ./manage.py help

    Particularly handy ones :

      ./manage.py shell_plus       ## import models from all installed apps 
  

EOU
}
djext-dir(){ echo $(local-base)/env/dj/django-command-extensions ; }
djext-cd(){  cd $(djext-dir); }
djext-mate(){ mate $(djext-dir) ; }
djext-get(){
   local dir=$(dirname $(djext-dir)) &&  mkdir -p $dir && cd $dir
   hg clone http://hgsvn.trbs.net/django-command-extensions
}

djext-ln(){
   python-
   python-ln $(djext-dir)/django_extensions
}

djext-build(){
  djext-get
  djext-ln
}


