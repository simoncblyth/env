# === func-gen- : doc/rst2pdf fgp doc/rst2pdf.bash fgn rst2pdf fgh doc
rst2pdf-src(){      echo doc/rst2pdf.bash ; }
rst2pdf-source(){   echo ${BASH_SOURCE:-$(env-home)/$(rst2pdf-src)} ; }
rst2pdf-vi(){       vi $(rst2pdf-source) ; }
rst2pdf-env(){      elocal- ; }
rst2pdf-usage(){ cat << EOU

RST2PDF
==========

* https://code.google.com/p/rst2pdf/
* http://rst2pdf.ralsina.com.ar/

Dependencies
--------------

* reportlab-



EOU
}

rst2pdf-dir(){ echo $(local-base)/env/doc/$(rst2pdf-name) ; }
rst2pdf-name(){ echo rst2pdf-0.93 ; }
rst2pdf-cd(){  cd $(rst2pdf-dir); }
rst2pdf-mate(){ mate $(rst2pdf-dir) ; }
rst2pdf-get(){
   local dir=$(dirname $(rst2pdf-dir)) &&  mkdir -p $dir && cd $dir

   local url=https://rst2pdf.googlecode.com/files/$(rst2pdf-name).tar.gz
   local tgz=$(basename $url)
   local nam=${tgz/.tar.gz}

   [ ! -f "$tgz" ] && curl -L -O $url 
   [ ! -d "$nam" ] && tar zxvf $tgz

}
