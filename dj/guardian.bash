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

     Good background description of django 1.2 underpinning of this 
         http://djangoadvent.com/1.2/object-permissions/ 


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


     On syncdb ...

Creating tables ...
Creating table guardian_userobjectpermission
Creating table guardian_groupobjectpermission
Installing custom SQL ...
Installing indexes ...
No fixtures found.



 == concerning the "fairview" fork ==

    *  When using "syncdb" from an fresh DB, gives MySQL operational error...
       while creating table : guardian_userobjectpermission
       "BLOB/TEXT column 'object_pk' used in key specification without a key length"

    * the admin.py that the fork adds is less than stellar ... clumsy interface 



EOU
}
guardian-dir(){ echo $(local-base)/env/dj/django-guardian/$(guardian-fork) ; }
guardian-cd(){  cd $(guardian-dir); }
guardian-mate(){ mate $(guardian-dir) ; }

guardian-fork(){
  echo ${GUARDIAN_FORK:-lukaszb}   ## master 
  #echo ${GUARDIAN_FORK:-fairview}   ## with admin
}

guardian-url(){ echo http://github.com/$(guardian-fork)/django-guardian.git ;  }
guardian-get(){
   local dir=$(dirname $(guardian-dir)) &&  mkdir -p $dir && cd $dir
   git clone $(guardian-url) $(guardian-fork) 
}


guardian-ln(){
   python-
   python-ln $(guardian-dir)/guardian
}


