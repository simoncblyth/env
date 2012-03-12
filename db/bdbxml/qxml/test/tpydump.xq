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

declare function my:dumper($arg as node()) as xs:string* external; 

for $rez in collection("dbxml:/hfc")//rez:quote[rez:qtag='BR:-531:-431,431*/BR:-511:-431,411']
return 
    my:dumper($rez)

