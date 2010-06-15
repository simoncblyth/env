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

     changes...  add "belle7..."
         {hosts, ["localhost","belle7.nuu.edu.tw"]}.



   == installations ==

    N : epel5 installation with yum   ejabberd 2.0.5 
          (2.0.5 was released 2009-04-03 ... apparently the last of the 2.0 series )




EOU
}
ejabberd-dir(){ echo $(local-base)/env/messaging/messaging-ejabberd ; }
ejabberd-cd(){  sudo su ; cd $(ejabberd-confdir); }
ejabberd-mate(){ mate $(ejabberd-dir) ; }
ejabberd-get(){
   local dir=$(dirname $(ejabberd-dir)) &&  mkdir -p $dir && cd $dir

}

ejabberd-confdir(){       echo /etc/ejabberd ; }
ejabberd-confpath(){      echo $(ejabberd-confdir)/ejabberd.cfg ; }
ejabberd-ctlconfpath(){   echo $(ejabberd-confdir)/ejabberdctl.cfg ; }
ejabberd-edit(){  sudo bash -c "cd $(ejabberd-confdir) ; vi $(ejabberd-confpath) $(ejabberd-ctlconfpath) ; " ; }


ejabberd-cookiepath(){ echo /var/lib/ejabberd/.erlang.cookie ; }
ejabberd-logdir(){     echo /var/log/ejabberd ; }
ejabberd-logpath(){    echo $(ejabberd-logdir)/ejabberd.log ; }
ejabberd-tail(){       tail -f $(ejabberd-logpath) ; }


ejabberd-status(){     sudo ejabberdctl status ; }
ejabberd-start(){      sudo ejabberdctl start ; }
ejabberd-stop(){       sudo ejabberdctl stop ; }

