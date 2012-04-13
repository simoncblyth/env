#!/usr/bin/env qxml -k k -v 511 -k map -v code2latex
(: 
  ./test-map.xq -k map -v code2latex -k k -v 511
:)
import module namespace my="http://my" at "my.xqm" ;
my:map($map, ($k) )  

