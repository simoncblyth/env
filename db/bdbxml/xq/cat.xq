#!/usr/bin/env qxml -k col -v dbxml:/sys -k nam -v qtaq2latex.xml
(:
   ./cat.xq -k nam -v qtag2svgs.xml
   ./cat.xq -k nam -v qtag2latex.xml
   
 this is failing when do not provide the nam argument    

let $col := "dbxml:/sys" 
let $nam := "qtag2latex.xml"
:)
let $doc := collection($col)/*[dbxml:metadata('dbxml:name')=$nam]
return 
       $doc/*




