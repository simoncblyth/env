# === func-gen- : messaging/pyxmpp fgp messaging/pyxmpp.bash fgn pyxmpp fgh messaging
pyxmpp-src(){      echo messaging/pyxmpp.bash ; }
pyxmpp-source(){   echo ${BASH_SOURCE:-$(env-home)/$(pyxmpp-src)} ; }
pyxmpp-vi(){       vi $(pyxmpp-source) ; }
pyxmpp-env(){      elocal- ; }
pyxmpp-usage(){
  cat << EOU
     pyxmpp-src : $(pyxmpp-src)
     pyxmpp-dir : $(pyxmpp-dir)


     http://pyxmpp.jajcus.net
     http://pyxmpp.jajcus.net/trac/

     Preqs ...
         Python 2.6
         libxml2 with dev headers  >=2.6.11 
         dnspython   http://www.dnspython.org/

EOU
}
pyxmpp-dir(){ echo $(local-base)/env/messaging/pyxmpp ; }
pyxmpp-cd(){  cd $(pyxmpp-dir); }
pyxmpp-mate(){ mate $(pyxmpp-dir) ; }
pyxmpp-get(){
   local dir=$(dirname $(pyxmpp-dir)) &&  mkdir -p $dir && cd $dir

   svn co http://pyxmpp.jajcus.net/svn/pyxmpp/trunk pyxmpp

}
