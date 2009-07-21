insulate-vi(){ vi $BASH_SOURCE ; }
insulate-usage(){
   package-fn  $FUNCNAME $*
   cat << EOU
   
      http://code.google.com/p/insulatenoseplugin/wiki/Documentation
   
     env-rsync nose 
     env-rsync python
  

     [blyth@cms01 NoseTests]$ nosetests -v --with-insulate 

 
EOU
}

insulate-env(){
  elocal-
  package-  
}


insulate-get(){

  easy_install -Z InsulateRunner

}

