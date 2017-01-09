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




EOU
}
rst-dir(){ echo $(local-base)/env/tools/rst/tools/rst-rst ; }
rst-cd(){  cd $(rst-dir); }
rst-mate(){ mate $(rst-dir) ; }
rst-get(){
   local dir=$(dirname $(rst-dir)) &&  mkdir -p $dir && cd $dir

}
