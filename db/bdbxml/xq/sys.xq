#!/usr/bin/env qxml 
(: list all docs 
:)

let $uri := "dbxml:/sys" 
for $a in collection($uri) return dbxml:metadata("dbxml:name", $a) 



