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


     TODO ...
         find way to avoid this dialog when starting from a fresh DB

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






