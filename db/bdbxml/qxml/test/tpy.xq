#!/usr/bin/env qxml.py -k color -v blue
(: 
   NB the **.py** this is using the same DBXML containers
   but a different controlling binary "qxml.py" rather than "qxml"

   Demos using external python function  
:)

declare function my:foo() as xs:string* external; 
(
   $color, 
   my:foo(),
   count(collection("dbxml:/hfc")//rez:rez)
)
