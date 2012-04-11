#!/usr/bin/env qxml -k code -v 81
(: 
    external function declarations are brought in together in the my.xqm module

   ./test-code2latex.xq -k code -v -211 -k code -v 211

       note the last key value pair is used when keys are repeated
       TODO:  make repeats yield a sequence,  $code = ( "-211", "211" )

:)
import module namespace my="http://my" at "my.xqm" ;
my:code2latex(("-11","11",$code))  

