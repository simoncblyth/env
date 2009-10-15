# === func-gen- : base/html fgp base/html.bash fgn html fgh base
html-src(){      echo base/html.bash ; }
html-source(){   echo ${BASH_SOURCE:-$(env-home)/$(html-src)} ; }
html-vi(){       vi $(html-source) ; }
html-env(){      elocal- ; }
html-usage(){
  cat << EOU
     html-src : $(html-src)
     html-dir : $(html-dir)


EOU
}
html-dir(){ echo $(env-home)/base ; }
html-cd(){  cd $(html-dir); }
html-mate(){ mate $(html-dir) ; }
html-href(){ python $(html-dir)/html.py $* ; }



