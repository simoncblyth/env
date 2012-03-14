#!/usr/bin/env qxml

for $a in collection() return dbxml:metadata("dbxml:name", $a)


