# === func-gen- : messaging/speeqe fgp messaging/speeqe.bash fgn speeqe fgh messaging
speeqe-src(){      echo messaging/speeqe.bash ; }
speeqe-source(){   echo ${BASH_SOURCE:-$(env-home)/$(speeqe-src)} ; }
speeqe-vi(){       vi $(speeqe-source) ; }
speeqe-env(){      elocal- ; }
speeqe-usage(){
  cat << EOU
     speeqe-src : $(speeqe-src)
     speeqe-dir : $(speeqe-dir)

      http://code.stanziq.com/speeqe
      http://code.stanziq.com/speeqe/wiki/SpeeqeSetup


    Getting many 504 Gateway Time-out ... is that normal for BOSH communication ?

      "timeouts" section of :
           http://tools.ietf.org/id/draft-loreto-http-bidirectional-02.txt

      http://www.checkupdown.com/status/E504.html


EOU
}
speeqe-dir(){ echo $(local-base)/env/messaging/speeqe ; }
speeqe-cd(){  cd $(speeqe-dir)/$1 ; }
speeqe-mate(){ mate $(speeqe-dir) ; }
speeqe-get(){
   local dir=$(dirname $(speeqe-dir)) &&  mkdir -p $dir && cd $dir
   git clone git://code.stanziq.com/speeqe
}


speeqe-build(){

   #speeqe-get
   nginx-
   nginx-ln $(speeqe-dir)/speeqewebclient

   local settings=$(speeqe-dir)/speeqewebclient/scripts/local_settings.js
   #cp $settings.dist $settings
   speeqe-conf > $settings 

}

speeqe-host(){ hostname ; }
speeqe-bind(){ echo /http-bind/ ; }


speeqe-conf(){ cat << EOC

//domain for your website
Speeqe.HTTP_DOMAIN = "$(speeqe-host)";
//domain for your jabber server
Speeqe.XMPP_DOMAIN = "$(speeqe-host)";
//domain for your multi user chat server
//multi user chat server name. This is the default chat server used if
//none is provided. If you connect to a room, the room will be
//room@Speeqe.CHAT_SERVER.
Speeqe.CHAT_SERVER = "$(speeqe-host)";
//allows users to use the /nick command to change their muc name
Speeqe.ENABLE_NICK_CHANGE = true;
//the default chat room if none is specified. If a muc room is not
//provided, and the user connects, this will be the default room.
Speeqe.DEFAULT_CHAT_ROOM = "speeqers@$(speeqe-host)";

//the url used to proxy to your BOSH server.  Used by speeqe to
//communicate with the bosh server.
Speeqe.BOSH_URL =  "$(speeqe-bind)";

//This is used to add additional help information to the help
//dialog. It will be displayed right before the instructions to close
//the dialog.
Speeqe.helpDialogHtml = "";


EOC
}

speeqe-nginx(){ cat << EOC

location ~ ^/(speeqewebclient/scripts)/ {
                # serve static files
            
            expires 0d;
            root   /usr/local/var/www/;
            index  index.html index.htm;
}


EOC
}


speeqe-statics(){

   local target=${1:-$(nginx-htdocs)}
   speeqe-cd speeqeweb/webroot

   sudo cp -r favicon.ico $target/
   sudo cp -r css         $target/
   sudo cp -r images      $target/
   sudo cp -r js          $target/

}


