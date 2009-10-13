# === func-gen- : jqplot/jqplot fgp jqplot/jqplot.bash fgn jqplot fgh jqplot
jqplot-src(){      echo jqplot/jqplot.bash ; }
jqplot-source(){   echo ${BASH_SOURCE:-$(env-home)/$(jqplot-src)} ; }
jqplot-vi(){       vi $(jqplot-source) ; }
jqplot-env(){      elocal- ; }
jqplot-usage(){
  cat << EOU
     jqplot-src : $(jqplot-src)
     jqplot-dir : $(jqplot-dir)


EOU
}
jqplot-dir(){ echo $(local-base)/env/jqplot ; }
jqplot-cd(){  cd $(jqplot-dir); }
jqplot-mate(){ mate $(jqplot-dir) ; }
jqplot-get(){
   local dir=$(dirname $(jqplot-dir)) &&  mkdir -p $dir && cd $dir
   hg clone http://bitbucket.org/cleonello/jqplot jqplot
}
