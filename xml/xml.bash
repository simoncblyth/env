

 xml_iwd=$(pwd) 
 
 ## caution must exit with initial dir  
 [ -t 0 ] || return 
 
 XML_BASE=$ENV_BASE/xml
 export XML_HOME=$HOME/$XML_BASE

 [ "$XML_DBG" == "1" ] && echo $XML_BASE/xml.bash

 cd $XML_HOME
 
 [ -r exist.bash ] && . exist.bash
 [ -r modjk.bash ] && . modjk.bash
 
 ## caution must exit with initial dir 
 cd $xml_iwd

