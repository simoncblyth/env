# === func-gen- : sa/formalchemy fgp sa/formalchemy.bash fgn formalchemy fgh sa
formalchemy-src(){      echo sa/formalchemy.bash ; }
formalchemy-source(){   echo ${BASH_SOURCE:-$(env-home)/$(formalchemy-src)} ; }
formalchemy-vi(){       vi $(formalchemy-source) ; }
formalchemy-env(){      elocal- ; }
formalchemy-usage(){
  cat << EOU
     formalchemy-src : $(formalchemy-src)
     formalchemy-dir : $(formalchemy-dir)


   * http://spyced.blogspot.com/2008/10/formalchemy-10.html
   * http://www.djangosnippets.org/snippets/1291/
   * http://wiki.pylonshq.com/display/pylonscookbook/Forms

 Dependency tempita :
    http://pythonpaste.org/tempita/


EOU
}
formalchemy-dir(){ echo $(local-base)/env/sa/formalchemy ; }
formalchemy-cd(){  cd $(formalchemy-dir); }
formalchemy-mate(){ mate $(formalchemy-dir) ; }
formalchemy-get(){
   local dir=$(dirname $(formalchemy-dir)) &&  mkdir -p $dir && cd $dir
   [ ! -d "formalchemy" ] && hg clone https://formalchemy.googlecode.com/hg/ formalchemy


   [ ! -d "tempita" ] && svn co http://svn.pythonpaste.org/Tempita/trunk/tempita 
   
   # /opt/local/bin/python2.5 `which hg`  clone https://formalchemy.googlecode.com/hg/ formalchemy
   # curl -L -O http://formalchemy.googlecode.com/files/FormAlchemy-1.2.tar.gz
   # svn checkout http://formalchemy.googlecode.com/svn/trunk/ formalchemy

  hg clone http://hg.python-rum.org/rum/

  hg clone http://toscawidgets.org/hg/ToscaWidgets/ 

}

formalchemy-ln(){
  python- ; 
  python-ln $(formalchemy-dir)/formalchemy ;
  python-ln $(dirname $(formalchemy-dir))/tempita ;
}

formalchemy-check(){

  python -c "import formalchemy "

  # http://www.sqlalchemy.org/trac/wiki/SqlSoup


}



