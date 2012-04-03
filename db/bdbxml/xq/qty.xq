#!/usr/bin/env qxml.py -k qty -v BR_-521_-431+111
(: 
single qty 
~~~~~~~~~~~

::
     qxml.py qty.xq -k qty -v BR_-521_-431+111
     qxml    qty.xq -k qty -v BR_-521_-431+111

:)

import module namespace rezu="http://hfag.phys.ntu.edu.tw/hfagc/rezu" at "rezu.xqm" ;
let $hfc := collection('dbxml:/hfc')
let $tty := rezu:iqt2name($qty)  
return ($qty,$tty, count($hfc//rez:quote[rez:qtag=$tty]), $hfc//rez:quote[rez:qtag=$tty] )


