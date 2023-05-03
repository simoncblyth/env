# === func-gen- : tools/rst/rst fgp tools/rst/rst.bash fgn rst fgh tools/rst
rst-src(){      echo tools/rst/rst.bash ; }
rst-source(){   echo ${BASH_SOURCE:-$(env-home)/$(rst-src)} ; }
rst-vi(){       vi $(rst-source) ; }
rst-env(){      elocal- ; }
rst-usage(){ cat << EOU

RST references
=================

Online RST renderer
-------------------

* http://www.tele3.cz/jbar/rest/rest.html

Spacing in RST
----------------


Links in RST
---------------
 
hello_ cruel_

.. _cruel: world
.. _hello: there



Compare raw and github rendered rst-cheatsheet 
------------------------------------------------

* https://raw.githubusercontent.com/ralsina/rst-cheatsheet/master/rst-cheatsheet.rst
* https://github.com/ralsina/rst-cheatsheet/blob/master/rst-cheatsheet.rst

* http://docutils.sourceforge.net/docs/ref/rst/restructuredtext.html

* https://github.com/ralsina/rst-cheatsheet/blob/master/rst-cheatsheet.rst


Anonymous Hyperlinks
----------------------

* http://docutils.sourceforge.net/docs/ref/rst/restructuredtext.html#anonymous-hyperlinks

Translation of tracwiki table leads to the Trac underscores coming thru and
confusing RST.

To debug::

    rst2pseudoxml.py Storage.rst Storage.pxml


EOU
}
rst-dir(){ echo $(local-base)/env/tools/rst ; }
rst-cd(){  cd $(rst-dir); }
rst-get(){
   local dir=$(rst-dir) &&  mkdir -p $dir && cd $dir
   
   local furl
   rst-url- | while read furl 
   do
       local url=$($furl) 
       printf "%20s : %s \n" $furl $url
 
       [ ! -f $(basename $url) ] && curl -L -O $url
   done
}

rst-url-(){ cat << EOU
rst-ref-url
rst-cheat-url
rst-directives-url
rst-rest-url
EOU
}

rst-cheat-url(){        echo https://raw.githubusercontent.com/ralsina/rst-cheatsheet/master/rst-cheatsheet.rst ; }
rst-ref-url(){          echo http://docutils.sourceforge.net/docs/ref/rst/restructuredtext.html ; }
rst-directives-url(){   echo http://docutils.sourceforge.net/docs/ref/rst/directives.html ; }
rst-rest-url(){         echo http://www.sphinx-doc.org/en/stable/rest.html ; }

rst-ref(){        open $(rst-dir)/$(basename $(rst-ref-url)) ; }
rst-cheat(){      vi   $(rst-dir)/$(basename $(rst-cheat-url)); }
rst-directives(){ open $(rst-dir)/$(basename $(rst-directives-url)); }
rst-rest(){       open $(rst-dir)/$(basename $(rst-rest-url)); }



