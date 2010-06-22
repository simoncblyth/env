# === func-gen- : messaging/strophe fgp messaging/strophe.bash fgn strophe fgh messaging
strophe-src(){      echo messaging/strophe.bash ; }
strophe-source(){   echo ${BASH_SOURCE:-$(env-home)/$(strophe-src)} ; }
strophe-vi(){       vi $(strophe-source) ; }
strophe-env(){      elocal- ; }
strophe-usage(){
  cat << EOU

     Strophe : 
       pure javascript library to facilitate creation of a javascript xmpp/jabber client 
       that talks over BOSH (Bidirectional-streams Over Synchronous HTTP) to an xmpp server
       such as ejabberd (which supports http-bind)

     http://code.stanziq.com/strophe/

       strophe API
          http://code.stanziq.com/strophe/strophejs/doc/1.0.1/files/core-js.html
    

     == echobot setup ==

       1) get and build Strophe   {{{strophe-;strophe-build}}}

           To build minified versions need to point to the jar with 
           YUI_COMPRESSOR 

       2) configure ejabberd + http-bind + nginx setup 
               http://anders.conbere.org/blog/2009/09/29/get_xmpp_-_bosh_working_with_ejabberd_firefox_and_strophe/
               http://gist.github.com/272956

          ejabberd-http-bind
              shows what needs to be added to nginx config... 
                 to proxy :xxxx/http-bind into standard port
         

     == debugging tips for Strophe http-bind connection ==

       Use Firebug javascript console while observing : ejabberd-tail 
{{{
xhr = new XMLHttpRequest()
xhr.open("POST", "/http-bind/" , true)
xhr.send(null)
}}}
   * gives a bad request 400 : no data ...   
   * BUT at least they are talking... demonstrating that "/http-bind/" is OK for the nginx/ejabberd config in use
   

    = strophe echobot status =

       ||   || Erlang   ||                                                   ||
       || C ||          || not working a termination occurs                  ||
       || N ||          || works .... can chat with the webpage from ichat   ||  


EOU
}
strophe-dir(){ echo $(local-base)/env/messaging/strophejs ; }
strophe-cd(){  cd $(strophe-dir); }
strophe-mate(){ mate $(strophe-dir) ; }
strophe-get(){
   local dir=$(dirname $(strophe-dir)) &&  mkdir -p $dir && cd $dir
   git clone http://github.com/metajack/strophejs.git
}

strophe-build(){

   strophe-get
   strophe-make
   strophe-ln

   strophe-echobot-conf 
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
strophe-echobot-conf(){
  perl -pi -e 's,/xmpp-httpbind,/http-bind/, ' $(strophe-dir)/examples/echobot.js
}
