#!/usr/bin/env qxml

let $grps:=collection('dbxml:/sys')/*[dbxml:metadata('dbxml:name')='v1qtags.xml']
return 
   for $group in $grps//group return data($group/@class)


