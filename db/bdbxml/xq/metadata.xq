#!/usr/bin/env qxml
(: 
#!/usr/bin/env qxml.py 
:)
declare function my:metadata($a as node()) as xs:double* external;
let $hfc := collection('dbxml:/hfc')
return 
for $n in $hfc
     return my:metadata($n) 


