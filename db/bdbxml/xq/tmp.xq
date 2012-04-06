#!/usr/bin/env qxml 
(:
for $a in collection("dbxml:/tmp") return dbxml:metadata("dbxml:name", $a) 
:)
for $a in collection("/tmp/hfagc/scratch.dbxml") return dbxml:metadata("dbxml:name", $a) 



