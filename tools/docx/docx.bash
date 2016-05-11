# === func-gen- : tools/docx/docx fgp tools/docx/docx.bash fgn docx fgh tools/docx
docx-src(){      echo tools/docx/docx.bash ; }
docx-source(){   echo ${BASH_SOURCE:-$(env-home)/$(docx-src)} ; }
docx-vi(){       vi $(docx-source) ; }
docx-env(){      elocal- ; }
docx-usage(){ cat << EOU

Python-docx
=============

python-docx is a Python library for creating 
and updating Microsoft Word (.docx) files.

* https://python-docx.readthedocs.org/en/latest/
* https://python-docx.readthedocs.io/en/latest/user/text.html
* https://python-docx.readthedocs.org/en/latest/user/styles-understanding.html

Hmm looks like macports drastically out of date
-------------------------------------------------

* https://github.com/mikemaccana/python-docx


Definition list support ?
---------------------------

Workaround by using bullet list instead.

::

    2016-05-08 09:43:52,691 env.doc.docutils.rst2docx INFO     reading ntu-report-may-2016.rst 
    NotImplementedError: env.doc.docutils.rst2docx.Translator visiting unknown node type: definition_list

Emphasis and Strong support 
------------------------------

Added by reworking the translation to docx.



See also
---------

* docxbuilder-


EOU
}
docx-dir(){  echo $(local-base)/env/tools/docx ; }
docx-sdir(){ echo $(env-home)/tools/docx ; }
docx-scd(){  cd $(docx-sdir); }
docx-cd(){   cd $(docx-dir); }

docx-get()
{
   local dir=$(dirname $(docx-dir)) &&  mkdir -p $dir && cd $dir

    git clone https://github.com/python-openxml/python-docx docx
}

docx-ln()
{
    python-
    python-ln $(docx-dir)/docx
}

docx-test()
{
    cd ~/workflow/admin/reps

    rst2docx.py test.rst /tmp/test.docx

    open /tmp/test.docx
}

