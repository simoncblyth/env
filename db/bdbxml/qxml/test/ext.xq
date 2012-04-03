#!/usr/bin/env qxml
(:
This continues to work even after a clean, unlike with qxml.py,
because the extfun symbols are `installed` into the env/bin/qxml binary 

Run this same XQuery using the C++ extensions via the pyextfun module using::

   make         # updates env/bin/qxml
   make pyx     # updates the pyextfun

   qxml.py test/ext.xq

   test/ext.xq           # shebang running
   qxml test/ext.xq      # same 

let $frag := element frag { "hello" }

:)
declare function my:pow($a as xs:double, $b as xs:double) as xs:double external;
declare function my:sqrt($a as xs:double) as xs:double external;
declare function my:quote2values($a as node()) as xs:double* external;


let $frag := <frag/> 

return 
(
my:pow(2,10),
my:sqrt(16),
my:sqrt(my:pow(2,10)),
my:quote2values($frag),
my:quote2values(<another/>),
my:quote2values(<another><with><sprogs/></with></another>),
$frag
)



