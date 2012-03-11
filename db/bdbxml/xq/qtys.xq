#!/usr/bin/env qxml -k code -v 01102
(: 

qtys within group
~~~~~~~~~~~~~~~~~~~

::
     ./qtys.xq -h
     ./qtys.xq -k code -v 01103

#. for container access need baseURI of dbxml:/ but for module import need -b ""
#. module loading only working from same directory 


cli parameters
~~~~~~~~~~~~~~~~

let $code  := '01102'

:)

import module namespace rezu="http://hfag.phys.ntu.edu.tw/hfagc/rezu" at "rezu.xqm" ;

let $grps  := collection('dbxml:/sys')/*[dbxml:metadata('dbxml:name')='v1qtags.xml']
let $group := $grps/*[@class=$code]
for $qtag in $group/qtag 
return 
   let $qty := data($qtag/@value)  
   let $tty := rezu:qt2name($qty)  
   return ($tty, count(collection('dbxml:/hfc')//rez:quote[rez:qtag=$qty]) )


