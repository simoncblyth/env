# === func-gen- : doc/rst2html/rst2html fgp doc/rst2html/rst2html.bash fgn rst2html fgh doc/rst2html
rst2html-src(){      echo doc/rst2html/rst2html.bash ; }
rst2html-source(){   echo ${BASH_SOURCE:-$(env-home)/$(rst2html-src)} ; }
rst2html-vi(){       vi $(rst2html-source) ; }
rst2html-env(){      elocal- ; docutils- ; }
rst2html-usage(){ cat << EOU

DOCUTILS STANDALONE RST2HTML
=================================

* http://docutils.sourceforge.net/docs/user/tools.html#rst2html-py

Stylesheet Customization
--------------------------

* http://docutils.sourceforge.net/docs/howto/html-stylesheets.html


EOU
}
rst2html-dir(){ echo $(local-base)/env/doc/rst2html/doc/rst2html-rst2html ; }
rst2html-cd(){  cd $(rst2html-dir); }
rst2html-mate(){ mate $(rst2html-dir) ; }
rst2html-get(){
   local dir=$(dirname $(rst2html-dir)) &&  mkdir -p $dir && cd $dir

}

rst2html-cssname(){ echo html4css1 ; }
rst2html-csspath(){ echo $(docutils-dir)/writers/$(rst2html-cssname)/$(rst2html-cssname).css ; }

rst2html-dest(){  echo $HOME/simoncblyth.bitbucket.org ; }

rst2html-csscopy(){
   local css=$(rst2html-csspath)
   [ ! -f "$css" ] && echo $msg NO css $css && return 

   local cmd="cp $(rst2html-csspath) $(rst2html-dest)/"
   echo $cmd
   eval $cmd

}

