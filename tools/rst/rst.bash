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

* https://github.com/ralsina/rst-cheatsheet/blob/master/rst-cheatsheet.rst



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
rst-refurl
rst-cheaturl
EOU
}

rst-cheaturl(){ echo https://raw.githubusercontent.com/ralsina/rst-cheatsheet/master/rst-cheatsheet.rst ; }
rst-refurl(){   echo http://docutils.sourceforge.net/docs/ref/rst/restructuredtext.html ; }

rst-ref(){ open $(rst-dir)/$(basename $(rst-refurl)) ; }
rst-cheat(){ vi $(rst-dir)/$(basename $(rst-cheaturl)); }

