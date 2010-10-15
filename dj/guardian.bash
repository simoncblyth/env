# === func-gen- : dj/guardian fgp dj/guardian.bash fgn guardian fgh dj
guardian-src(){      echo dj/guardian.bash ; }
guardian-source(){   echo ${BASH_SOURCE:-$(env-home)/$(guardian-src)} ; }
guardian-vi(){       vi $(guardian-source) ; }
guardian-env(){      elocal- ; }
guardian-usage(){
  cat << EOU
     guardian-src : $(guardian-src)
     guardian-dir : $(guardian-dir)

     http://packages.python.org/django-guardian/
         an implementation of object permissions for Django 1.2 via an extra authentication backend

     http://packages.python.org/django-guardian/configuration.html    

     Usage :

        guardian-get
        guardian-ln

     In django 1.2+ settings.py of project add 3 things and run syncdb

         INSTALLED_APPS = (
               # ...
               'guardian',
         )

         AUTHENTICATION_BACKENDS = (
             'django.contrib.auth.backends.ModelBackend', # this is default
             'guardian.backends.ObjectPermissionBackend',
         )

         ANONYMOUS_USER_ID = -1






EOU
}
guardian-dir(){ echo $(local-base)/env/dj/django-guardian ; }
guardian-cd(){  cd $(guardian-dir); }
guardian-mate(){ mate $(guardian-dir) ; }
guardian-url(){ echo http://github.com/lukaszb/django-guardian.git ;  }

guardian-get(){
   local dir=$(dirname $(guardian-dir)) &&  mkdir -p $dir && cd $dir
   git clone $(guardian-url) 
}

guardian-ln(){
   python-
   python-ln $(guardian-dir)/guardian
}


