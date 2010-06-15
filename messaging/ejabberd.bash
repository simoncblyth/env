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

   == configuration ==

     config is loaded from .cfg file into mnesia database at 1st start,
     subseqents starts ignore the .cfg file ... cfg can be changed from
     web interface (but this is not reflected back into the .cfg file) 

     config changes...  documented in wiki:Ejabberd

   == installations ==

    N : epel5 installation with yum   ejabberd 2.0.5 
          (2.0.5 was released 2009-04-03 ... apparently the last of the 2.0 series )


EOU
}
ejabberd-dir(){ echo $(local-base)/env/messaging/ejabberd ; }
ejabberd-cd(){  cd $(ejabberd-confdir); }
ejabberd-mate(){ mate $(ejabberd-dir) ; }
ejabberd-get(){
   local dir=$(dirname $(ejabberd-dir)) &&  mkdir -p $dir && cd $dir
   echo $msg ... on N installed with yum from epel5 ... on C not available in epel4 so ...

   if [ ! -d ejabberd ]; then
      git clone git://git.process-one.net/ejabberd/mainline.git ejabberd
      cd ejabberd
      git checkout -b 2.0.x origin/2.0.x
   else
      echo $msg ejabberd dir already exists 
   fi

}

ejabberd-confdir(){       echo /etc/ejabberd ; }
ejabberd-confpath(){      echo $(ejabberd-confdir)/ejabberd.cfg ; }
ejabberd-ctlconfpath(){   echo $(ejabberd-confdir)/ejabberdctl.cfg ; }
ejabberd-edit(){  sudo vi $(ejabberd-confpath) $(ejabberd-ctlconfpath) ; }

ejabberd-ebin(){       echo /usr/lib/ejabberd/ebin ; }
ejabberd-include(){    echo /usr/lib/ejabberd/include ; }
ejabberd-cookie(){     echo /var/lib/ejabberd/.erlang.cookie ; }
ejabberd-logdir(){     echo /var/log/ejabberd ; }
ejabberd-logpath(){    echo $(ejabberd-logdir)/ejabberd.log ; }
ejabberd-log(){       sudo vi $(ejabberd-logpath) ; }
ejabberd-tail(){      sudo tail -f $(ejabberd-logpath) ; }

ejabberd-ctl(){        sudo ejabberdctl $* ; }
ejabberd-status(){     ejabberd-ctl status ; }
ejabberd-start(){      ejabberd-ctl start ; }
ejabberd-stop(){       ejabberd-ctl stop ; }

ejabberd-connected(){     ejabberd-ctl connected-users ; }

ejabberd-open(){    
   iptables-
   IPTABLES_PORT=$(local-port ejabberd) iptables-webopen  
}
ejabberd-webadmin(){     
   local cmd="curl http://localhost:$(local-port ejabberd-http)/admin" 
   echo $msg $cmd
   eval $cmd
}
ejabberd-register-(){
   local pfx="${1:-}"
   private-
   local cmd="sudo ejabberdctl register $(private-val EJABBERD_USER$pfx) $(private-val EJABBERD_HOST$pfx) $(private-val EJABBERD_PASS$pfx) "
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
