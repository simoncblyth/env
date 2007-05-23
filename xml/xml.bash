 
 
 [ -t 0 ] || return 
 
 XML_BASE=$ENV_BASE/xml
 export XML_HOME=$HOME/$XML_BASE

 iwd=$(pwd) 
 cd $XML_HOME
 
 [ -r exist.bash ] && . exist.bash
 [ -r modjk.bash ] && . modjk.bash
 [ -r workflow.bash ] && . workflow.bash
 
 cd $iwd

