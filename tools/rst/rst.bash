# === func-gen- : tools/rst/rst fgp tools/rst/rst.bash fgn rst fgh tools/rst
rst-src(){      echo tools/rst/rst.bash ; }
rst-source(){   echo ${BASH_SOURCE:-$(env-home)/$(rst-src)} ; }
rst-vi(){       vi $(rst-source) ; }
rst-env(){      elocal- ; }
rst-usage(){ cat << EOU

RST references
=================

Compare raw and github rendered rst-cheatsheet 
------------------------------------------------

* https://raw.githubusercontent.com/ralsina/rst-cheatsheet/master/rst-cheatsheet.rst
* https://github.com/ralsina/rst-cheatsheet/blob/master/rst-cheatsheet.rst

* http://docutils.sourceforge.net/docs/ref/rst/restructuredtext.html



EOU
}
rst-dir(){ echo $(local-base)/env/tools/rst ; }
rst-cd(){  cd $(rst-dir); }
rst-get(){
   local dir=$(rst-dir) &&  mkdir -p $dir && cd $dir
   [ ! -f restructuredtext.html ] && curl -L -O $(rst-refurl)
}

rst-refurl(){ echo http://docutils.sourceforge.net/docs/ref/rst/restructuredtext.html ; }
rst-ref(){ open $(rst-dir)/restructuredtext.html ; }

