



nosenose-(){   . $ENV_HOME/nose/package/nosenose.bash && nosenose-env $* ; } 
xmlnose-(){    . $ENV_HOME/nose/package/xmlnose.bash  && xmlnose-env  $* ; } 


nose-env(){
  elocal-
  package-
}