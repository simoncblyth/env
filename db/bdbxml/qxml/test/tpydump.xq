#!/usr/bin/env qxml.py 
(: 
NB using **qxml.py** not qxml, but talking to same DB XML containers
rather than ``qxml``::

      ./tpydump.xq -l DEBUG

Demos using an external python extension function ``my:dumper`` 
:)

declare function my:dumper($arg as node()) as xs:string* external; 

for $rez in collection("dbxml:/hfc")//rez:quote[rez:qtag='BR:-531:-431,431*/BR:-511:-431,411']
return 
    my:dumper($rez)

