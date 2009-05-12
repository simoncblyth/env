# === func-gen- : xml/xmldiff.bash fgp xml/xmldiff.bash fgn xmldiff
xmldiff-src(){      echo xml/xmldiff.bash ; }
xmldiff-source(){   echo ${BASH_SOURCE:-$(env-home)/$(xmldiff-src)} ; }
xmldiff-vi(){       vi $(xmldiff-source) ; }
xmldiff-env(){      elocal- ; }
xmldiff-usage(){
  cat << EOU
     xmldiff-src : $(xmldiff-src)

EOU
}

xmldiff-dir(){ echo /tmp/env/$FUNCNAME ; }
xmldiff-nam(){ echo xmldiff-0.6.9 ; }
xmldiff-url(){ echo http://ftp.logilab.org/pub/xmldiff/$(xmldiff-nam).tar.gz ; }
xmldiff-get(){
   local f=${FUNCNAME/-*}
   local dir=$($f-dir) && mkdir -p $dir
   local iwd=$PWD

   cd $dir
   local url=$($f-url)
   local nam=$($f-nam)
   local tgz=$(basename $url)
   [ ! -f "$tgz" ] && curl -O $url 
   [ ! -d "$nam" ] && tar zxvf $tgz   

   #cd $iwd 
}

