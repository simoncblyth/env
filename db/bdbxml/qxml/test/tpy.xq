#!/usr/bin/env qxml.py -k color -v blue -k hello -v world 
(: 
NB the **.py** on the shebang line, this is using the 
same DBXML containers but a different controlling main ``qxml.py`` 
rather than ``qxml``

Override default color above with::

      ./tpy.xq -k color -v red 
      ./tpy.xq -k color -v red -k hello -v will-not-propagate

The variables are added to the query running context, but these fail
to override vars like $hello assigned in the code. 

Demos using external python function  

:)

declare function my:foo() as xs:string* external; 

let $hello := "sorry overriding you" 
return 
(
   $color, 
   $hello, 
   my:foo(),
   count(collection("dbxml:/hfc")//rez:rez)
)
