#!/usr/bin/env qxml -k code -v 01102
(: 

Handles provide opaque node reference within container
container identification depends on default prefixes such as "dbxml:////" 
so it seems that need to omit that


observations
~~~~~~~~~~~~

#. ``document-uri root smth``  does not work, in more involved locations .. suspect a steps removed effect
   dbxml:metadata("dbxml:name",$q) 

           document-uri(root($qtag))



                           $q/dbxml:metadata("dbxml:name"), 
                           dbxml:metadata("dbxml:name",$q) 


    for $q in $quo 
                  return 
                           $q/dbxml:metadata("dbxml:name"),
                           $q/dbxml:node-to-handle()

         let $quo := collection('dbxml:/hfc')//rez:quote[rez:qtag=$qty]
      dbxml:handle-to-node('/tmp/hfagc/hfagc.dbxml','BiMCAp8AzA=='),
         let $tty := rezu:qt2name($qty)  

:)
import module namespace rezu="http://hfag.phys.ntu.edu.tw/hfagc/rezu" at "rezu.xqm" ;
let $grps  := collection('dbxml:/sys')/*[dbxml:metadata('dbxml:name')='v1qtags.xml']
return 
   (
     for $qtag at $iq in $grps/*[@class=$code]/qtag 
     return 
     ( 
         let $qty := data($qtag/@value)  
         let $hdl := collection('dbxml:/hfc')//rez:quote[rez:qtag=$qty]/dbxml:node-to-handle()
         return ($iq,$hdl)
      )

   )

