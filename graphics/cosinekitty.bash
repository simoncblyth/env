# === func-gen- : graphics/cosinekitty fgp graphics/cosinekitty.bash fgn cosinekitty fgh graphics
cosinekitty-src(){      echo graphics/cosinekitty.bash ; }
cosinekitty-source(){   echo ${BASH_SOURCE:-$(env-home)/$(cosinekitty-src)} ; }
cosinekitty-vi(){       vi $(cosinekitty-source) ; }
cosinekitty-env(){      elocal- ; }
cosinekitty-usage(){ cat << EOU


* http://www.cosinekitty.com/raytrace/raytrace_a4.pdf




EOU
}
cosinekitty-pdf(){ open $(local-base)/env/graphics/cosinekitty/raytrace_a4.pdf ; }
cosinekitty-dir(){ echo $(local-base)/env/graphics/cosinekitty/raytrace ; }
cosinekitty-cd(){  cd $(cosinekitty-dir); }
cosinekitty-c(){  cd $(cosinekitty-dir)/$1 ; }
cosinekitty-mate(){ mate $(cosinekitty-dir) ; }
cosinekitty-get(){
   local dir=$(dirname $(cosinekitty-dir)) &&  mkdir -p $dir && cd $dir

   local url=http://cosinekitty.com/raytrace/rtsource.zip
   local nam=$(basename $url) 

   [ ! -f $nam ] && curl -L -O $url 

   [ ! -d raytrace ] && unzip $nam

   [ ! -d raytrace_a4.pdf ] && curl -L -O http://www.cosinekitty.com/raytrace/raytrace_a4.pdf

}
