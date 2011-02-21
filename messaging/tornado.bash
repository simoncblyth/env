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
   tornado-build

      builds 'tornado.epoll' extension

   tornado-install 



   == install attempts ==

       system yum py24 on N 
 
         sudo yum --enablerepo=epel install pycurl
         sudo yum --enablerepo=epel install python-simplejson

            ... nope lots of py2.6 isms "with" ... encountered at tornado-install step

       port py26 on G ...   simplejson not needed as "json" comes standard with py26

            virtualenv $HOME/v/mq 
            cd ; . v/mq/bin/activate
            pip install pycurl        


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
   python setup.py install   ## no need for sudo as using virtualenv python on G 
}

tornado-helloworld(){
   local msg="=== $FUNCNAME :"
   echo $msg test with : curl http://localhost:8888
   python $(tornado-dir)/demos/helloworld/helloworld.py
}
