module namespace my="http://my";
declare function my:code2latex($a as xs:string*) as xs:string* external;
declare function my:metadata($a as node()) as xs:double* external;
