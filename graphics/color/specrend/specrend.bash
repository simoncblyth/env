# === func-gen- : graphics/color/specrend fgp graphics/color/specrend.bash fgn specrend fgh graphics/color
specrend-src(){      echo graphics/color/specrend/specrend.bash ; }
specrend-source(){   echo ${BASH_SOURCE:-$(env-home)/$(specrend-src)} ; }
specrend-vi(){       vi $(specrend-source) ; }
specrend-env(){      elocal- ; }
specrend-usage(){ cat << EOU

SPECREND
========

From wavelength to RGB values

* http://www.fourmilab.ch/documents/specrend/

Others
-------

* http://codingmess.blogspot.tw/2009/05/conversion-of-wavelength-in-nanometers.html


EOU
}
specrend-dir(){ echo $(env-home)/graphics/color/specrend ; }
specrend-cd(){  cd $(specrend-dir); }
specrend-mate(){ mate $(specrend-dir) ; }
specrend-get(){
   mkdir -p $(specrend-dir)
   specrend-cd
   local url=http://www.fourmilab.ch/documents/specrend/specrend.c
   local nam=$(basename $url)
   [ ! -f "$nam" ] && curl -L -O $url
}
