#!/usr/bin/env qxml
(: 
   TODO:
      adjust quote2values to take multiple quotes just like "my:avg" will
:)

import module namespace rezu="http://hfag.phys.ntu.edu.tw/hfagc/rezu" at "rezu.xqm" ;
declare function my:quote2values($a as node()) as xs:double* external;

let $nam := "babar/cecilia/b0dspi.xml"
let $rez := collection('dbxml:/hfc')/*[dbxml:metadata('dbxml:name')=$nam]

return ($nam, for $q in $rez/rez:data/rez:quote return my:quote2values($q) ) 


