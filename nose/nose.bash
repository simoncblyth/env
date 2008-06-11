



nosenose-(){   . $ENV_HOME/nose/package/nosenose.bash && nosenose-env $* ; } 
xmlnose-(){    . $ENV_HOME/nose/package/xmlnose.bash  && xmlnose-env  $* ; } 

_xmlnose(){    python $ENV_HOME/xmlnose/main.py --with-xml-output $* ; }



nose-env(){
  elocal-
  package-
}