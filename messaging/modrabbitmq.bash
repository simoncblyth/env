# === func-gen- : messaging/modrabbitmq fgp messaging/modrabbitmq.bash fgn modrabbitmq fgh messaging
modrabbitmq-src(){      echo messaging/modrabbitmq.bash ; }
modrabbitmq-source(){   echo ${BASH_SOURCE:-$(env-home)/$(modrabbitmq-src)} ; }
modrabbitmq-vi(){       vi $(modrabbitmq-source) ; }
modrabbitmq-env(){      elocal- ; }
modrabbitmq-usage(){
  cat << EOU
     modrabbitmq-src : $(modrabbitmq-src)
     modrabbitmq-dir : $(modrabbitmq-dir)

     ejabberd extension that bridges to rabbitmq

        http://ndpar.blogspot.com/2010/03/integrating-rabbitmq-with-ejabberd.html


    


EOU
}

modrabbitmq-base(){  echo $(dirname $(modrabbitmq-dir)) ; }
modrabbitmq-dir(){ echo $(local-base)/env/messaging/modrabbitmq ; }
modrabbitmq-cd(){  cd $(modrabbitmq-base); }
modrabbitmq-mate(){ mate $(modrabbitmq-dir) ; }
modrabbitmq-get(){
   local dir=$(dirname $(modrabbitmq-dir)) &&  mkdir -p $dir && cd $dir

   if [ ! -d ejabberd ]; then
      git clone git://git.process-one.net/ejabberd/mainline.git ejabberd
      cd ejabberd
      git checkout -b 2.0.x origin/2.0.x
   fi

   cd $dir
   if [ ! -d rabbitmq-xmpp ]; then
      hg clone http://hg.rabbitmq.com/rabbitmq-xmpp
      cd rabbitqmq-xmpp

      ## take a punt on a compatible version  
      #hg up pre_switch_to_non_embedding   ... no .hrl at this version 
      hg up tip
   fi
}

modrabbitmq-kbuild(){
   cd $(modrabbitmq-base)

   cp rabbitmq-xmpp/src/mod_rabbitmq.erl ejabberd/src/ 
   cp rabbitmq-xmpp/src/rabbit.hrl       ejabberd/src/
   
   cd ejabberd/src
   ./configure --disable-tls
   make
}

modrabbitmq-kinstall(){

   cd $(modrabbitmq-base)/ejabberd/src
 
   ejabberd-
   sudo cp mod_rabbitmq.beam $(ejabberd-ebin)/

   ## hmm not promising that have to create this !!!
   sudo mkdir -p $(ejabberd-include) 
   sudo cp rabbit.hrl        $(ejabberd-include)/
}

modrabbitmq-ls(){

   ejabberd-
   rabbitmq-

   sudo ls -l $(ejabberd-ebin)/mod_rabbitmq.beam
   sudo ls -l $(ejabberd-include)/rabbit.hrl
   sudo ls -l $(ejabberd-cookie)
   sudo ls -l $(rabbitmq-cookie) 

}

modrabbitmq-config(){  cat << EOC
   
   with ejabberd-edit added the below to the modules section ...
       {mod_rabbitmq, [{rabbitmq_node, rabbit@belle7}]},
          
EOC
}

modrabbitmq-cookie-align(){

   ejabberd-
   rabbitmq-

   local ans 
   read -p "$msg ejabberd-stop before you do this ... enter YES to proceed " ans
   [ "$ans" != "YES" ] && echo $msg skipping && return 
  
   sudo mv $(ejabberd-cookie) $(ejabberd-cookie).orig
   sudo cp $(rabbitmq-cookie) $(ejabberd-cookie)
   sudo chown ejabberd:ejabberd  $(ejabberd-cookie)
}

modrabbitmq-cookie-ls(){
   sudo ls -l $(ejabberd-cookie)
   sudo cat $(ejabberd-cookie)
   echo
   sudo ls -l $(rabbitmq-cookie)
   sudo cat $(rabbitmq-cookie) 
   echo
}

