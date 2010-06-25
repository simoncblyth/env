# === func-gen- : messaging/xmpppy fgp messaging/xmpppy.bash fgn xmpppy fgh messaging
xmpppy-src(){      echo messaging/xmpppy.bash ; }
xmpppy-source(){   echo ${BASH_SOURCE:-$(env-home)/$(xmpppy-src)} ; }
xmpppy-vi(){       vi $(xmpppy-source) ; }
xmpppy-env(){      elocal- ; }
xmpppy-usage(){
  cat << EOU
     xmpppy-src : $(xmpppy-src)
     xmpppy-dir : $(xmpppy-dir)

     http://xmpppy.sourceforge.net/


EOU
}
xmpppy-dir(){ echo $(local-base)/env/messaging/xmpppy ; }
xmpppy-cd(){  cd $(xmpppy-dir); }
xmpppy-mate(){ mate $(xmpppy-dir) ; }
xmpppy-get(){
   local dir=$(dirname $(xmpppy-dir)) &&  mkdir -p $dir && cd $dir

   cvs -d:pserver:anonymous@xmpppy.cvs.sourceforge.net:/cvsroot/xmpppy login 

   echo $msg hit return when prompted for password
   cvs -z3 -d:pserver:anonymous@xmpppy.cvs.sourceforge.net:/cvsroot/xmpppy co xmpppy  
}

xmpppy-ln(){
   python-
   python-ln $(xmpppy-dir)/xmpp
}

xmpppy-send(){    python $(env-home)/messaging/xmpppy/send.py $* ; }


