# === func-gen- : html/html fgp html/html.bash fgn html fgh html
html-src(){      echo html/html.bash ; }
html-source(){   echo ${BASH_SOURCE:-$(env-home)/$(html-src)} ; }
html-vi(){       vi $(html-source) ; }
html-env(){      elocal- ; }
html-usage(){
  cat << EOU
     html-src : $(html-src)
     html-dir : $(html-dir)

     http://www.decalage.info/python/html



EOU
}
html-dir(){ echo $(local-base)/env/html/$(html-name) ; }
html-cd(){  cd $(html-dir); }
html-mate(){ mate $(html-dir) ; }
html-get(){
   local dir=$(dirname $(html-dir)) &&  mkdir -p $dir && cd $dir

   [ ! -f "$(html-name).zip" ] && curl -O $(html-url)
   [ ! -d "$(html-name)"  ]    && unzip $(html-name).zip
}

html-name(){ echo HTML.py-0.04 ;  }
html-url(){  echo http://www.decalage.info/files/$(html-name).zip ; }


html-install(){
   html-cd
   sudo python setup.py install
}

html-build(){

   html-get
   html-install

}

