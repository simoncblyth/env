nose-vi(){ vi $BASH_SOURCE ; }
nose-usage(){

   cat << EOU

   hmm these are downloading into 
      eg /usr/local/env/trac/package/insulate 
   
EOU


}


insulate-(){   . $ENV_HOME/nose/package/insulate.bash && insulate-env $* ; } 
nosenose-(){   . $ENV_HOME/nose/package/nosenose.bash && nosenose-env $* ; } 
xmlnose-(){    . $ENV_HOME/nose/package/xmlnose.bash  && xmlnose-env  $* ; } 

_xmlnose(){    python $ENV_HOME/xmlnose/main.py --with-xml-output $* ; }



nose-env(){
  elocal-
  package-
}
