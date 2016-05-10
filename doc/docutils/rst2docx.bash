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



EOU
}
rst2docx-dir(){ echo $(env-home)/doc/docutils ; }
rst2docx-cd(){  cd $(rst2docx-dir); }
