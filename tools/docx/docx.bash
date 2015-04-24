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

* https://python-docx.readthedocs.org/en/latest/user/styles-understanding.html


Hmm looks like macports drastically out of date
-------------------------------------------------

* https://github.com/mikemaccana/python-docx

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



