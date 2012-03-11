#!/usr/bin/env qxml -k uri -v dbxml:/hfc -k color -v blue
(: list all docs ....  aliases are hfc and sys

   ./ls.xq  -k uri -v dbxml:/hfc
   ./ls.xq  -k uri -v dbxml:/hfc -k color -v red
   ./ls.xq  -k uri -v dbxml:/sys -k color -v red

Parameterized to some extent: 

#. commandline correctly trumps shebang line above 
#. but variable declaration in .xq trumps the parameters, so are forced to not include default variables in code

let $uri := "dbxml:/sys" 
:)


for $a in collection($uri) return dbxml:metadata("dbxml:name", $a) 



