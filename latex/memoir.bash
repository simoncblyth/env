# === func-gen- : latex/memoir fgp latex/memoir.bash fgn memoir fgh latex
memoir-src(){      echo latex/memoir.bash ; }
memoir-source(){   echo ${BASH_SOURCE:-$(env-home)/$(memoir-src)} ; }
memoir-vi(){       vi $(memoir-source) ; }
memoir-env(){      elocal- ; 
   export TEXINPUTS=$(memoir-dir):
}
memoir-usage(){
  cat << EOU
     memoir-src : $(memoir-src)
     memoir-dir : $(memoir-dir)


      http://www.tex.ac.uk/cgi-bin/texfaq2html?label=tempinst


EOU
}
memoir-dir(){ echo $(local-base)/env/latex/memoir ; }
memoir-cd(){  cd $(memoir-dir); }
memoir-mate(){ mate $(memoir-dir) ; }

memoir-url(){
   echo http://mirror.ctan.org/macros/latex/contrib/memoir.zip
}

memoir-get(){
   local dir=$(dirname $(memoir-dir)) &&  mkdir -p $dir && cd $dir
   [ ! -f "memoir.zip" ] && curl -L -O $(memoir-url)
   [ ! -d "memoir" ] && unzip memoir.zip 
}

memoir-build(){
    memoir-cd
    latex memoir.ins

}


