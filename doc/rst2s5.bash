# === func-gen- : doc/rst2s5 fgp doc/rst2s5.bash fgn rst2s5 fgh doc
rst2s5-src(){      echo doc/rst2s5.bash ; }
rst2s5-source(){   echo ${BASH_SOURCE:-$(env-home)/$(rst2s5-src)} ; }
rst2s5-vi(){       vi $(rst2s5-source) ; }
rst2s5-env(){      elocal- ; }
rst2s5-usage(){ cat << EOU

RST2S5
=======

See also::

    simon:e blyth$ ls -l ~/e/doc/rst2s5
    total 64
    -rw-r--r--  1 blyth  staff  25327 13 Nov 12:53 slide-shows.txt
    -rw-r--r--  1 blyth  staff   3316 13 Nov 12:53 test_rst2s5_py.txt


* http://docutils.sourceforge.net/docs/user/slide-shows.html
* :google:`s5 themes`

  * http://meyerweb.com/eric/tools/s5/themes/



::

    simon:Documents blyth$ which rst2s5-2.6.py
    /opt/local/bin/rst2s5-2.6.py

    simon:Documents blyth$ rst2s5-2.6.py --help
    Usage
    =====
      rst2s5-2.6.py [options] [<source> [<destination>]]

    Generates S5 (X)HTML slideshow documents from standalone reStructuredText
    sources.  Reads from <source> (default is stdin) and writes to <destination>
    (default is stdout).  See <http://docutils.sf.net/docs/user/config.html> for
    the full reference.


RST Roles
----------

* http://docutils.sourceforge.net/docs/ref/rst/roles.html
* http://docutils.sourceforge.net/docs/howto/rst-roles.html



Themes
-------

::

    simon:rst2s5 blyth$ rst2s5-2.6.py --theme small-white slide-shows.txt  slide-shows-sw.html
    simon:rst2s5 blyth$ open  slide-shows-sw.html
    simon:rst2s5 blyth$ rst2s5-2.6.py --theme small-black slide-shows.txt  slide-shows-sb.html
    simon:rst2s5 blyth$ open  slide-shows-sb.html


FUNCTIONS
-----------

*rst2s5-help*
      many many options

*rst2s5-quickstart*
      copy in Makefile and initial 


EOU
}
rst2s5-dir(){ echo $(local-base)/env/doc/rst2s5 ; }
rst2s5-cd(){  cd $(rst2s5-dir); }
rst2s5-get(){
  local dir=$(rst2s5-dir) &&  mkdir -p $dir && cd $dir
  echo the script comes along with docutils : this just gets the example slide show document
  local url=http://docutils.sourceforge.net/docs/user/slide-shows.txt
  local name=$(basename $url)
  [ ! -f "$name" ] && curl -O  $url
}

rst2s5(){ rst2s5-2.6.py $* ; }
rst2s5-help(){ rst2s5 --help ; }
rst2s5-quickstart(){
   local msg="=== $FUNCNAME :"

   [ ! -f Makefile ] && cp $(rst2s5-dir)/Makefile .
   [ ! -f slide-shows.txt.template ]  && cp $(rst2s5-dir)/slide-shows.txt  slide-shows.txt.template 

   echo $msg get started by : cp slide-shows.txt.template name.txt  then \"make\" 
}





