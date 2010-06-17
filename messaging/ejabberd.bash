# === func-gen- : messaging/ejabberd fgp messaging/ejabberd.bash fgn ejabberd fgh messaging
ejabberd-src(){      echo messaging/ejabberd.bash ; }
ejabberd-source(){   echo ${BASH_SOURCE:-$(env-home)/$(ejabberd-src)} ; }
ejabberd-vi(){       vi $(ejabberd-source) ; }
ejabberd-env(){      elocal- ; }
ejabberd-usage(){
  cat << EOU
     ejabberd-src : $(ejabberd-src)
     ejabberd-dir : $(ejabberd-dir)

      bug reports 
        http://support.process-one.net/

      official docs
        http://www.process-one.net/en/ejabberd/docs/

      community site
        http://www.ejabberd.im/

      my investigations
        wiki:Ejabberd

   == funcs ==

   commandline check the webadmin is accessible ..
      ejabberd-webadmin | xmllint --format - 


   == configuration ==

     config is loaded from .cfg file into mnesia database at 1st start,
     subseqents starts ignore the .cfg file ... cfg can be changed from
     web interface (but this is not reflected back into the .cfg file) 

     config changes...  documented in wiki:Ejabberd



   == versions ==


   ||        ||   ejabberd   ||   rabbitmq-server   ||    rabbitmq-xmpp    ||
   ||  N yum ||   2.0.5      ||    1.7.2            ||  56:0be11e0cbd86    ||            
   ||  N src ||   2.1.x      ||    1.7.2            ||  56:0be11e0cbd86    ||            
   ||  C     ||   2.1.x      ||    1.7.0            ||  56:0be11e0cbd86    ||               

     Partially working on C ... not working on N .. 
         ... curious would suspect the other way around ... am i sure using top of -xmpp

     http://hg.rabbitmq.com/rabbitmq-server/tags

       1.7.0 ... Wed Oct 07 14:47:05 2009 +0100 (8 months ago) .... http://hg.rabbitmq.com/rabbitmq-server/rev/b1089fcc31b7
       1.7.2 ... Tue Feb 16 12:03:04 2010 +0000 (4 months ago) ...  http://hg.rabbitmq.com/rabbitmq-server/rev/bacb333d7645 
      

   == installations ==

    N : epel5 installation with yum   ejabberd 2.0.5 
          (2.0.5 was released 2009-04-03 ... apparently the last of the 2.0 series )

    C : not available in epel4 ... so source install
   

   == C : source install ==

      http://www.process-one.net/en/ejabberd/guide_en#htoc8

   == issues ==

   === on starting/restarting ejabberd(with mod_rabbitmq) on C ... see error in rabbitmq-tail ===

{{{
=ERROR REPORT==== 16-Jun-2010::15:06:31 ===
** Connection attempt from disallowed node ejabberd@localhost ** 
}}}

     After ejabberd-cookie-align ... rabbitmq-tail says :
{{{
=INFO REPORT==== 16-Jun-2010::15:27:28 ===
node ejabberd@localhost up
}}}



  === rabbitmq-xmpp ===

 [blyth@cms01 rabbitmq-xmpp]$ rpm -ql rabbitmq-server | grep .hrl
/usr/lib/rabbitmq/lib/rabbitmq_server-1.7.0/include/rabbit.hrl
/usr/lib/rabbitmq/lib/rabbitmq_server-1.7.0/include/rabbit_framing.hrl
/usr/lib/rabbitmq/lib/rabbitmq_server-1.7.0/include/rabbit_framing_spec.hrl
[blyth@cms01 rabbitmq-xmpp]$ 
[blyth@cms01 rabbitmq-xmpp]$ diff /usr/lib/rabbitmq/lib/rabbitmq_server-1.7.0/include/rabbit.hrl src/rabbit.hrl
21c21

   some diffs but not disastrous


EOU
}
ejabberd-base(){ echo $(dirname $(ejabberd-dir)) ; }
ejabberd-dir(){ echo $(local-base)/env/messaging/ejabberd ; }
ejabberd-cd(){  cd $(ejabberd-dir); }
ejabberd-mate(){ mate $(ejabberd-dir) ; }

ejabberd-info(){
    yum --enablerepo=epel info ejabberd
}

ejabberd-wipe(){
  cd $(ejabberd-base)
  rm -rf ejabberd
  rm -rf rabbitmq-xmpp 
}

ejabberd-get(){
   local dir=$(ejabberd-base) &&  mkdir -p $dir && cd $dir
   echo $msg ... on N installed with yum from epel5 ... on C not available in epel4 so ...

   if [ ! -d ejabberd ]; then
      git clone git://git.process-one.net/ejabberd/mainline.git ejabberd
   else
      echo $msg ejabberd dir already exists 
   fi
   cd ejabberd
   #git checkout -b 2.0.x origin/2.0.x
   git checkout -b 2.1.x origin/2.1.x

   cd $dir
   if [ ! -d rabbitmq-xmpp ]; then
      hg clone http://hg.rabbitmq.com/rabbitmq-xmpp
   else
      echo rabbitmq-xmpp already exists
   fi 
   cd rabbitmq-xmpp
   hg up tip
}

ejabberd-cf(){
   rabbitmq-
   local cmd="diff $(rabbitmq-hrl) $(ejabberd-base)/rabbitmq-xmpp/src/rabbit.hrl "
   echo $msg $cmd
   eval $cmd
}

ejabberd-rabbit-copyin(){


   cd $(ejabberd-base)/rabbitmq-xmpp

   ## hg up bafcc3d61adb
   ## go back to :
   ##        http://hg.rabbitmq.com/rabbitmq-xmpp/rev/bafcc3d61adb    
   ##            Thu Aug 27 16:21:24 2009 +0100 (9 months ago)
   ##     
   ##  which is contemporary with tag 1.7.0 of the server :
   ##         http://hg.rabbitmq.com/rabbitmq-server/rev/b1089fcc31b7
   ##              Wed Oct 07 14:47:05 2009 +0100 (8 months ago)
   ##
   ##   BUT thats on a different branch ... 
   ##       pick the next revision on the right branch ...
   ##       and bingo starting to work somewhat ...
   ##       ... but sending binaries (eg with rootmq-sendobj) kills the connection 
   ##       ... only raw string working 
   ##          
   hg up dd2dc489730f

   cp $(ejabberd-base)/rabbitmq-xmpp/src/mod_rabbitmq.erl $(ejabberd-dir)/src/   
   
   #echo $msg using the supplied rabbit.hrl from rabbitmq-xmpp
   #cp $(ejabberd-base)/rabbitmq-xmpp/src/rabbit.hrl       $(ejabberd-dir)/src/

   echo $msg try to use the canonical rabbit ...    
   rabbitmq-
   cp $(rabbitmq-hrl)  $(ejabberd-dir)/src/
} 


ejabberd-rabbit-diff(){
   diff $(ejabberd-base)/rabbitmq-xmpp/src/mod_rabbitmq.erl $(ejabberd-dir)/src/mod_rabbitmq.erl   
   diff $(ejabberd-base)/rabbitmq-xmpp/src/rabbit.hrl       $(ejabberd-dir)/src/rabbit.hrl   
}


ejabberd-configure(){
   ejabberd-cd
   ejabberd-rabbit-copyin
   cd src
   ./configure --prefix=$(ejabberd-prefix)  --exec-prefix=$(ejabberd-eprefix)
}

ejabberd-make(){
   ejabberd-cd
   cd src
   make
}

ejabberd-install(){
   ejabberd-cd
   cd src
   sudo make install

   ## oops looses originals ...  on 2nd pass 
   sudo cp $(ejabberd-confpath) $(ejabberd-confpath).original   
   sudo cp $(ejabberd-ctlconfpath) $(ejabberd-ctlconfpath).original

   ejabberd-cookie-align
   ejabberd-cookie-ls
}


ejabberd-sysuser(){ 
  if [ "$(ejabberd-prefix)" == "" ]; then
     echo ejabberd 
  else
     echo root
  fi
}
ejabberd-sysgroup(){  ejabberd-sysuser ; }

ejabberd-cookie-align(){

   rabbitmq-
   local ans
   read -p "$msg ejabberd-stop before you proceed with this ... rabbitmq cookie stays the same so no need to stop it  ... enter YES to proceed " ans
   [ "$ans" != "YES" ] && echo $msg skipping && return

   sudo mv $(ejabberd-cookie) $(ejabberd-cookie).original
   sudo cp $(rabbitmq-cookie) $(ejabberd-cookie)
   sudo chown $(ejabberd-sysuser):$(ejabberd-sysgroup)  $(ejabberd-cookie)
}

ejabberd-cookie-ls(){
   sudo ls -l $(ejabberd-cookie)
   sudo cat $(ejabberd-cookie)
   echo
   sudo ls -l $(rabbitmq-cookie)
   sudo cat $(rabbitmq-cookie)
   echo
}


ejabberd-build(){

   ejabberd-get
   ejabberd-configure
   ejabberd-make
   ejabberd-install

}

ejabberd-diff(){        sudo diff $(ejabberd-confpath).original $(ejabberd-confpath) ; } 
ejabberd-shost(){  hostname -s ; }

ejabberd-conf(){
   ## this localhost needs to be removed ... to avoid startup crash dump
   sudo perl -pi -e "s/^({hosts, \[\"localhost\")(\]}\.)\$/\$1,\"$(ejabberd-host)\"\$2/ " $(ejabberd-confpath)
   sudo perl -pi -e "s/^{loglevel, \d}.$/{loglevel, 5}./ "  $(ejabberd-confpath)
   sudo perl -pi -e "s/%%{acl, admin, {user, \"ermine\", \"example.org\"}}./{acl, admin, {user, \"$(ejabberd-user)\", \"$(ejabberd-host)\"}}./ " $(ejabberd-confpath)
   sudo perl -pi -e "s/(  {mod_vcard,    \[\]},)$/\$1\n  {mod_rabbitmq, [{rabbitmq_node, rabbit\@$(ejabberd-shost)}]},/  " $(ejabberd-confpath) 

   ejabberd-diff
}


ejabberd-prefix(){
 case $(hostname -s) in 
    cms01) echo $(local-base)/env/ejd ;;
   belle7) echo $(local-base)/env/ejd ;;
        *) echo -n ;;
 esac
}
ejabberd-eprefix(){     echo $(ejabberd-prefix)/usr ; }

ejabberd-confdir(){     echo $(ejabberd-prefix)/etc/ejabberd ; }
ejabberd-confpath(){    echo $(ejabberd-confdir)/ejabberd.cfg ; }
ejabberd-ctlconfpath(){ echo $(ejabberd-confdir)/ejabberdctl.cfg ; }

ejabberd-ebin(){        echo $(ejabberd-eprefix)/lib/ejabberd/ebin ; }
ejabberd-include(){     echo $(ejabberd-eprefix)/lib/ejabberd/include ; }
ejabberd-cookie(){      echo $(ejabberd-prefix)/var/lib/ejabberd/.erlang.cookie ; }
ejabberd-logdir(){      echo $(ejabberd-prefix)/var/log/ejabberd ; }

ejabberd-slogpath(){    
  if [ "$(ejabberd-prefix)" == "" ]; then
      echo $(ejabberd-logdir)/sasl.log  
  else
      echo $(ejabberd-logdir)/erlang.log 
  fi
}

ejabberd-logpath(){     echo $(ejabberd-logdir)/ejabberd.log ; }
ejabberd-logwipe(){   sudo rm $(ejabberd-logpath) $(ejabberd-slogpath) ; }

ejabberd-edit(){      sudo vi $(ejabberd-confpath) $(ejabberd-ctlconfpath) ; }
ejabberd-editctl(){   sudo vi $(ejabberd-ctlconfpath) ; }
ejabberd-log(){       sudo vi $(ejabberd-logpath) ; }
ejabberd-slog(){      sudo vi $(ejabberd-slogpath) ; }
ejabberd-tail(){      sudo tail -f $(ejabberd-logpath) ; }
ejabberd-stail(){     sudo tail -f $(ejabberd-slogpath) ; }

ejabberd-ctl(){        sudo $(ejabberd-prefix)/usr/sbin/ejabberdctl $* ; }
ejabberd-status(){     ejabberd-ctl status ; }
ejabberd-start(){      ejabberd-ctl start ; }
ejabberd-stop(){       ejabberd-ctl stop ; }

ejabberd-connected(){     ejabberd-ctl connected-users ; }

ejabberd-open(){    
   iptables-
   IPTABLES_PORT=$(local-port ejabberd) iptables-webopen  
}

ejabberd-webadmin-setup(){
   echo register user 0 ... needed for webadmin to work ... ejabberd has to be running for this to work 
   ejabberd-register- _0
   ejabberd-webadmin | xmllint --format -
}

ejabberd-webadmin-open(){    
   iptables-
   IPTABLES_PORT=$(local-port ejabberd-http) iptables-webopen 
}

ejabberd-user(){ echo $(private-;private-val EJABBERD_USER_0) ; }
ejabberd-host(){ echo $(private-;private-val EJABBERD_HOST_0) ; }
ejabberd-pass(){ echo $(private-;private-val EJABBERD_PASS_0) ; }


ejabberd-webadmin-creds(){ echo $(ejabberd-user)@$(ejabberd-host):$(ejabberd-pass) ; }
ejabberd-webadmin(){     
   local cmd="curl --anyauth --user $(ejabberd-webadmin-creds) http://localhost:$(local-port ejabberd-http)/admin" 
   #echo $msg $cmd
   eval $cmd
}
ejabberd-register-(){
   local pfx="${1:-}"
   private-
   local cmd="ejabberd-ctl register $(private-val EJABBERD_USER$pfx) $(private-val EJABBERD_HOST$pfx) $(private-val EJABBERD_PASS$pfx) "
   echo $cmd
   eval $cmd
}
ejabberd-register(){
   private-
   
   ## attempts to register the same name/host again... gets already registered warning 
   local ids="_1 _2 _3 _4 _5"
   for id in $ids ; do
      ejabberd-register- $id
   done
}
