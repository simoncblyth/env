#!/usr/bin/env qxml
(: 
#!/usr/bin/env qxml.py 

Usage example::

   ./mmetadata.xq | xmllint --format -
   ./mmetadata.xq > out.xml
   head -c 1000 out.xml      ## too much without newlines for vi
   tail -c 1000 out.xml

For valid XML requires discipline:

#. ``cout`` query result output
#. ``clog`` logging
#. ``cerr`` error reporting

:)
declare function my:mmetadata($a as node()*) as node() external;
let $hfc := collection('dbxml:/hfc')
return my:mmetadata($hfc)


