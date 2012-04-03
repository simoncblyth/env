
module namespace rezu="http://hfag.phys.ntu.edu.tw/hfagc/rezu" ;


declare variable $rezu:qt2n :=  (":/*,","_ox+")  ;

declare function rezu:qt2name(   $qt as xs:string )  as xs:string { translate( $qt  , $rezu:qt2n[1], $rezu:qt2n[2] ) } ;
declare function rezu:iqt2name(  $qt as xs:string )  as xs:string { translate( $qt  , $rezu:qt2n[2], $rezu:qt2n[1] ) } ;

declare function rezu:hello() as xs:string { "hello" } ;


