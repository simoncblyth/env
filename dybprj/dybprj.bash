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


dybprj-usage(){
  cat << EOU
     dybprj-src : $(dybprj-src)
     dybprj-dir : $(dybprj-dir)

     NB following dybprj- precursor the DJANGO_* "context" is switched to dybprj 
        providing generic django functionality via :
              
              dj- 
              djdep-      eg fastcgi deployment 

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
dybprj-cd(){  cd $(dybprj-dir); }
dybprj-mate(){ mate $(dybprj-dir) ; }


dybprj-dev-caution(){ cat << EOC

    ONLY DO THIS WHEN THE PORT IS PROTECTED ...
    AS EXPOSING A DEBUG ENABLED DJANGO IS AKIN TO EXPOSING YOUR SOUL

EOC
}

dybprj-dev(){
    local msg="=== $FUNCNAME :"
    dybprj-dev-caution
    dybprj-cd
    local cmd="./manage.py runserver $(hostname):$(dj-port)"
    echo $msg $cmd
    eval $cmd
}



