#!/usr/bin/env qxml
(: 
#!/usr/bin/env qxml.py 
#!/usr/bin/env qxml.py -k qty -v BR_-521_-431+111

single qty 
~~~~~~~~~~~

::
     qxml.py qty.xq -k qty -v BR_-521_-431+111
     qxml    qty.xq -k qty -v BR_-521_-431+111

:)

import module namespace rezu="http://hfag.phys.ntu.edu.tw/hfagc/rezu" at "rezu.xqm" ;
declare function my:quote2values($a as node()) as xs:double* external;

let $qty := "BR_-521_-431+111"
let $hfc := collection('dbxml:/hfc')
let $tty := rezu:iqt2name($qty)  
let $num := count($hfc//rez:quote[rez:qtag=$tty])
let $one := ($hfc//rez:quote[rez:qtag=$tty])[1] 

return ($qty,$tty,$num,$one, my:quote2values($one) ) 


