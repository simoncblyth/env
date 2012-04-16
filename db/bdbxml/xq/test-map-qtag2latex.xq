#!/usr/bin/env qxml -k k -v Azero:97:94,20443 -k map -v qtag2latex
(: 
  ./test-map.xq -k map -v qtag2latex -k k -v 511
:)
import module namespace my="http://my" at "my.xqm" ;
my:map($map, ($k) )  

