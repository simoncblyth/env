# === func-gen- : doc/docutils/docutils fgp doc/docutils/docutils.bash fgn docutils fgh doc/docutils
docutils-src(){      echo doc/docutils/docutils.bash ; }
docutils-source(){   echo ${BASH_SOURCE:-$(env-home)/$(docutils-src)} ; }
docutils-vi(){       vi $(docutils-source) ; }
docutils-env(){      elocal- ; }
docutils-usage(){ cat << EOU

DOCUTILS
==========



EOU
}
docutils-dir(){ python -c "import os, docutils ; print os.path.dirname(docutils.__file__)" ; }
docutils-cd(){  cd $(docutils-dir); }


docutils-sdir(){ echo $(env-home)/doc/docutils ; }
docutils-scd(){ cd $(docutils-sdir) ; }



