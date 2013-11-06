# === func-gen- : doc/sphinxhtmlslide fgp doc/sphinxhtmlslide.bash fgn sphinxhtmlslide fgh doc
sphinxhtmlslide-src(){      echo doc/sphinxhtmlslide.bash ; }
sphinxhtmlslide-source(){   echo ${BASH_SOURCE:-$(env-home)/$(sphinxhtmlslide-src)} ; }
sphinxhtmlslide-vi(){       vi $(sphinxhtmlslide-source) ; }
sphinxhtmlslide-env(){      elocal- ; }
sphinxhtmlslide-usage(){ cat << EOU





EOU
}
sphinxhtmlslide-dir(){ echo $(local-base)/env/doc/sphinxjp.themes.htmlslide ; }
sphinxhtmlslide-cd(){  cd $(sphinxhtmlslide-dir); }
sphinxhtmlslide-mate(){ mate $(sphinxhtmlslide-dir) ; }
sphinxhtmlslide-get(){
   local dir=$(dirname $(sphinxhtmlslide-dir)) &&  mkdir -p $dir && cd $dir

   hg clone https://bitbucket.org/shimizukawa/sphinxjp.themes.htmlslide

}
