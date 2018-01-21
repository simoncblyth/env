# === func-gen- : doc/docutils/docutils fgp doc/docutils/docutils.bash fgn docutils fgh doc/docutils
docutils-src(){      echo doc/docutils/docutils.bash ; }
docutils-source(){   echo ${BASH_SOURCE:-$(env-home)/$(docutils-src)} ; }
docutils-vi(){       vi $(docutils-source) ; }
docutils-env(){      elocal- ; }
docutils-usage(){ cat << EOU

DOCUTILS
==========


rst2odt.py
-----------

Translates RST into a Open Document Format .odt file

* http://docutils.sourceforge.net/docs/user/odt.html

rst2doc
--------

* https://github.com/trevorld/utilities/blob/master/bin/rst2doc

::

    base=`basename $1 .rst`
    rst2odt ${base}.rst ${base}.odt
    libreoffice --headless --convert-to doc ${base}.odt

rst2docx.py 
-------------

* https://github.com/python-openxml/python-docx

My development based on docutils and python-docx to 
convert simple RST reports into docx equivalents, using 
the default styles of python-docx which are fortunately 
neutral.

issue : table of contents
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://groups.google.com/forum/#!topic/python-docx/VnHD7AwmPgY
* http://stackoverflow.com/questions/18595864/python-create-a-table-of-contents-with-python-docx-lxml

.. a rendered ToC depends on pagination to know what page number to put for each
heading. Pagination is a function provided by the layout engine, a very complex
piece of software built into the Word client.


* https://github.com/python-openxml/python-docx/issues/36


plain text problem
-------------------

Why is the world still:

* sourcing its documents using bloated binary formats ?
* not using text based sources for all documents 
* not using version control for all documents 

* http://bettermess.com/the-plain-text-problem/



EOU
}

docutils-dir-svn(){ echo $LOCAL_BASE/env/doc/docutils ; }
docutils-get-svn(){
   local dir=$(dirname $(docutils-dir-svn))
   mkdir -p $dir 
   cd $dir
   [ ! -d docutils ] && svn checkout http://svn.code.sf.net/p/docutils/code/trunk docutils
}
docutils-cd-svn()
{
   cd $(docutils-dir-svn)
}



docutils-dir(){ python -c "import os, docutils ; print os.path.dirname(docutils.__file__)" ; }
docutils-cd(){  cd $(docutils-dir); }

docutils-find(){
  docutils-cd 
  pwd
  find . -name '*.py' -exec grep -H ${1:-Unknown\ interpreted\ text\ role} {} \;
}


docutils-sdir(){ echo $(env-home)/doc/docutils ; }
docutils-scd(){ cd $(docutils-sdir) ; }


docutils-rst2docx(){
   docutils-scd
   python rst2docx.py /tmp/report.rst /tmp/report.docx

}
