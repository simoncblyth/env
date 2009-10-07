# === func-gen- : pl/plbook fgp pl/plbook.bash fgn plbook fgh pl
plbook-src(){      echo pl/plbook.bash ; }
plbook-source(){   echo ${BASH_SOURCE:-$(env-home)/$(plbook-src)} ; }
plbook-vi(){       vi $(plbook-source) ; }
plbook-env(){      elocal- ; }
plbook-usage(){
  cat << EOU
     plbook-src : $(plbook-src)
     plbook-dir : $(plbook-dir)

     Could pull out generics into  sphinx- 

EOU
}

plbook-dir(){  echo $(local-base $*)/env/PylonsBook ; }
plbook-cd(){   cd $(plbook-dir) ; }
plbook-get(){
   local msg="=== $FUNCNAME :"
   local dir=$(dirname $(plbook-dir))
   mkdir -p $dir && cd $dir
   local nam=$(basename $(plbook-dir))
   [ ! -d "$nam" ] && hg clone https://hg.3aims.com/public/PylonsBook/ || echo $msg $nam is already cloned 
}
plbook-builddir(){ echo .build ; }
plbook-build(){
    plbook-cd
    sphinx-build -b html . ./$(plbook-builddir)
}
plbook-open(){
   local name=${1:-index.html}
   open file://$(plbook-dir)/$(plbook-builddir)/$name
}
plbook-build-latex(){
    plbook-cd
    sphinx-build -b latex . ./$(plbook-builddir)
}
plbook-build-pdf(){
    plbook-build-latex
    cd ./$(plbook-builddir)
    pdflatex PylonsBook.tex
    pdflatex PylonsBook.tex
}
plbook-open-pdf(){
   plbook-open PylonsBook.pdf
}




