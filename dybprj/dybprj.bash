# === func-gen- : dybprj/dybprj fgp dybprj/dybprj.bash fgn dybprj fgh dybprj
dybprj-src(){      echo dybprj/dybprj.bash ; }
dybprj-source(){   echo ${BASH_SOURCE:-$(env-home)/$(dybprj-src)} ; }
dybprj-vi(){       vi $(dybprj-source) ; }
dybprj-env(){      
   elocal- 
   export DJANGO_PROJECT=dybprj
   export DJANGO_APP=dbi
   export DJANGO_PROJDIR=$(dybprj-dir)
   export DJANGO_PROTOCOL=fcgi

   dj-
}

dybprj-ln(){
   python-
   python-ln $(dybprj-dir)
}


dybprj-usage(){
  cat << EOU
     dybprj-src : $(dybprj-src)
     dybprj-dir : $(dybprj-dir)

     NB following dybprj- precursor the DJANGO_* "context" is switched to dybprj 
        providing generic django functionality via :
              
              dj- 
              djdep-      eg fastcgi deployment 

     == CSRF issues ==

        On N was getting CSRF permission denied on submitting comment forms 
           * fixed by upgrading to trunk django and using RequestContext on all views


     == for dev running behind a protected port ONLY ==

          iptables-open $(dj-port)


     TODO ...
         find way to avoid this dialog when starting from a fresh DB
         (probably via fixture bootstrapping?)

    You just installed Django's auth system, which means you don't have any superusers defined.
    Would you like to create one now? (yes/no): yes
    Username (Leave blank to use 'blyth'): admin
    E-mail address: blyth@hep1.phys.ntu.edu.tw
    Password: 
    Password (again): 
    Superuser created successfully.

EOU
}
dybprj-dir(){ echo $(env-home)/dybprj ; }
dybprj-cd(){  cd $(dybprj-dir)/$1 ; }
dybprj-mate(){ mate $(dybprj-dir) ; }
dybprj-sh(){ dybprj-cd ; ./manage.py shell_plus ; }

dybprj-run-caution(){ cat << EOC

    ONLY DO THIS WHEN THE PORT IS PROTECTED ...
    AS EXPOSING A DEBUG ENABLED DJANGO IS AKIN TO EXPOSING YOUR SOUL

    NB live updating requires rabbitjs-run also 

EOC
}

dybprj-run(){
    local msg="=== $FUNCNAME :"
    dybprj-run-caution
    dybprj-cd
    local cmd="./manage.py runserver $(hostname):$(dj-port)"
    echo $msg $cmd
    eval $cmd
}



