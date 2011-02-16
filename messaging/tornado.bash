# === func-gen- : messaging/tornado fgp messaging/tornado.bash fgn tornado fgh messaging
tornado-src(){      echo messaging/tornado.bash ; }
tornado-source(){   echo ${BASH_SOURCE:-$(env-home)/$(tornado-src)} ; }
tornado-vi(){       vi $(tornado-source) ; }
tornado-env(){      elocal- ; }
tornado-usage(){
  cat << EOU
     tornado-src : $(tornado-src)
     tornado-dir : $(tornado-dir)

     http://www.tornadoweb.org/documentation

   tornado-check
 
      Pre-requisites with py2.4 on N  
         sudo yum --enablerepo=epel install pycurl
         sudo yum --enablerepo=epel install python-simplejson

   tornado-build

      builds 'tornado.epoll' extension

   tornado-install 
       errors from lots of python 2.6 isms ... "with"

EOU
}
tornado-dir(){ echo $(local-base)/env/messaging/$(tornado-name) ; }
tornado-cd(){  cd $(tornado-dir); }
tornado-mate(){ mate $(tornado-dir) ; }
tornado-name(){ echo tornado-1.1.1 ; }
tornado-url(){ echo http://github.com/downloads/facebook/tornado/$(tornado-name).tar.gz ; }
tornado-get(){
   local dir=$(dirname $(tornado-dir)) &&  mkdir -p $dir && cd $dir

   [ ! -f "$(tornado-name).tar.gz" ]  && curl -L -O $(tornado-url) 
   [ ! -d "$(tornado-name)" ]         && tar zxvf $(tornado-name).tar.gz

}


tornado-check(){
   python -c "import simplejson"
   python -c "import pycurl"
}

tornado-build(){
   tornado-cd
   python setup.py build
}

tornado-install(){
   type $FUNCNAME
   tornado-cd
   sudo python setup.py install
}


