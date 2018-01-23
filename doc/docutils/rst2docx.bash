# === func-gen- : doc/docutils/rst2docx fgp doc/docutils/rst2docx.bash fgn rst2docx fgh doc/docutils
rst2docx-src(){      echo doc/docutils/rst2docx.bash ; }
rst2docx-source(){   echo ${BASH_SOURCE:-$(env-home)/$(rst2docx-src)} ; }
rst2docx-vi(){       vi $(rst2docx-source) ; }
rst2docx-env(){      elocal- ; }
rst2docx-usage(){ cat << EOU

RST2DOCX : Translate RST to Word XML 
======================================

Simple translation of RST using docutils node tree
with a translator that contructs a python docx document.


See also

* docx-
* https://python-docx.readthedocs.io/en/latest/user/text.html


Typical Usage
---------------

::

    rst2docx.py index.rst /tmp/test.docx
    open /tmp/test.docx    
    # then print from Pages to make .pdf


Adding blockquote translation ?
----------------------------------

::

    2017-04-26 13:23:47,226 env.doc.docutils.rst2docx INFO     reading ntu-report-may-2017.rst 
    2017-04-26 13:23:47,396 env.doc.docutils.rst2docx INFO     Writer pre walkabout
    NotImplementedError: env.doc.docutils.rst2docx.Translator visiting unknown node type: block_quote
    Exiting due to error.  Use "--traceback" to diagnose.
    Please report errors to <docutils-users@lists.sf.net>.





EOU
}
rst2docx-dir(){ echo $(env-home)/doc/docutils ; }
rst2docx-cd(){  cd $(rst2docx-dir); }
