# === func-gen- : root/rootd fgp root/rootd.bash fgn rootd
rootd-src(){      echo root/rootd.bash ; }
rootd-source(){   echo ${BASH_SOURCE:-$(env-home)/$(rootd-src)} ; }
rootd-vi(){       vi $(rootd-source) ; }
rootd-env(){      elocal- ; }
rootd-usage(){
  cat << EOU
     rootd-src : $(rootd-src)


     Following instructions from :
           http://root.cern.ch/root/NetFile.html


EOU
}

rootd-user(){ echo ${ROOTD_USER:-blyth} ; }
rootd-xconf-(){  cat << EOC
# default: on
# description: The rootd daemon allows remote access to ROOT files.
service rootd
{
        disable = no
        socket_type             = stream
        wait                    = no
        user                    = $(rootd-user) 
        server                  = $(which rootd)
        server_args             = -r -i $(root-rootsys)
        log_on_success          += DURATION USERID
        log_on_failure          += USERID
}
EOC
}

rootd-xconf(){
  local msg="=== $FUNCNAME :"
  local tmp=/tmp/env/$FUNCNAME/rootd && mkdir -p $(dirname $tmp)
  $FUNCNAME- > $tmp
  echo $msg created $tmp
  cat $tmp
  echo  
  local cmd="sudo cp $tmp /etc/xinetd.d/rootd"
  echo $cmd
  eval $cmd
}

rootd-ps(){
    ps aux | grep rootd
}

rootd-conf(){
   local msg="=== $FUNCNAME :"
   ! grep  rootd /etc/services > /dev/null  && echo $msg ABORT no \"rootd 1094/tcp\" entry  in /etc/services && return 1
   rootd-xconf
}

rootd-xhup(){
   local msg="=== $FUNCNAME :"
   local xpid=$(cat /var/run/xinetd.pid 2>/dev/null)
   [ "$xpid" == "" ] && echo $msg no xinetd pid ? && return 1
   ps up $xpid
   local cmd="sudo kill -HUP $xpid"
   echo $msg $cmd
   eval $cmd
}

