#!/usr/bin/env qxml.py
declare function my:foo() as xs:string* external; 
declare function my:pow($a as xs:double, $b as xs:double) as xs:double external;
declare function my:sqrt($a as xs:double) as xs:double external;
(
my:foo(),
my:pow(2,10),
my:sqrt(16),
my:sqrt(my:pow(2,10))
)



