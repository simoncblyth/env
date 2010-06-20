# === func-gen- : messaging/strophe fgp messaging/strophe.bash fgn strophe fgh messaging
strophe-src(){      echo messaging/strophe.bash ; }
strophe-source(){   echo ${BASH_SOURCE:-$(env-home)/$(strophe-src)} ; }
strophe-vi(){       vi $(strophe-source) ; }
strophe-env(){      elocal- ; }
strophe-usage(){
  cat << EOU
     strophe-src : $(strophe-src)
     strophe-dir : $(strophe-dir)

     http://code.stanziq.com/strophe/

     To build minified versions need to point to the jar with 
     YUI_COMPRESSOR 




     Getting ejabberd + http-bind + nginx setup 
        http://anders.conbere.org/blog/2009/09/29/get_xmpp_-_bosh_working_with_ejabberd_firefox_and_strophe/

         * nginx 
         * ejabberd-http-bind
                ejabberd with http-bind proxied into standard port
         
    Getting strophe echobot example working      
         http://gist.github.com/272956

    Strophe echobot wants to talk with ...
         var BOSH_SERVICE = '/xmpp-httpbind'


EOU
}
strophe-dir(){ echo $(local-base)/env/messaging/strophejs ; }
strophe-cd(){  cd $(strophe-dir); }
strophe-mate(){ mate $(strophe-dir) ; }
strophe-get(){
   local dir=$(dirname $(strophe-dir)) &&  mkdir -p $dir && cd $dir
   git clone http://github.com/metajack/strophejs.git
}

strophe-make(){
   strophe-cd
   make
}

strophe-ln(){
  nginx-
  nginx-ln $(strophe-dir)
}

strophe-host(){ echo localhost ; }
strophe-echobot(){
  curl http://$(strophe-host)/strophejs/examples/echobot.html
}


