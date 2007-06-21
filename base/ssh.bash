
#
#   machines that are currently resisting arrest:
#       pal@nuu "L"
#

SSH_BASE=".ssh"


ssh--keygen(){

  local passph=${1:-dummy}
  [ "$passph" == "dummy" ] && echo "you must enter a passphrase as the argument " && return 
  [ -d "$HOME/$SSH_BASE" ] || ( mkdir $HOME/$SSH_BASE && chmod 700 $HOME/$SSH_BASE )

  echo generating keys on node $NODE_TAG
  local types="rsa1 rsa dsa"
  for typ in $types
  do	  
     case $typ in
       rsa1) keyname=identity ;;
	   rsa)  keyname=id_rsa   ;;
	   dsa)  keyname=id_dsa   ;;
	     *)  keyname=error    ;;
	  esac    
	  keyfile="$HOME/$SSH_BASE/$keyname"
	  if [ -f "$keyfile" ]; then
		  echo keyfile $keyfile already exists 
	  else	  
      
          echo ssh-keygen -t $typ -f $keyfile  -C "$typ from $NODE_TAG "  -N $passph
               ssh-keygen -t $typ -f $keyfile  -C "$typ from $NODE_TAG "  -N $passph   
	  fi	  
  done		
}

ssh--info(){

    local info=$(ssh--infofile)
    local fpid
    if [ -f "$info" ]; then
       fpid=$(perl -lne 'm/SSH_AGENT_PID=(\d*)/ && print $1' $info)
    else
       fpid=0
    fi
   
   if [ "X$SSH_AGENT_PID" == "X" ]; then
      echo SSH_AGENT_PID is not defined
   else
      if [ "$SSH_AGENT_PID" == "$fpid" ]; then
         echo SSH_AGENT_PID is defined and matches that of $info  
      else
         echo SSH_AGENT_PID $SSH_AGENT_PID does not match fpid $fpid 
      fi 
   fi
}



ssh--infofile(){
  echo $SSH_INFOFILE
}

ssh--agent-ok(){
  ## agent must be running and hold some identities to be "ok"
   (ssh-add -l >& /dev/null) && echo 1 || echo 0
}


ssh--agent-start(){
    local info=$(ssh--infofile)
    ssh-agent > $info && perl -pi -e 's/echo/#echo/' $info && chmod 0600 $info 
    
    echo ===== sourcing the info for the agent $info
    . $info
    
    echo ===== adding identities to the agent 
    
    if ([ -f $HOME/.ssh/identity ] && [ -f $HOME/.ssh/id_dsa ] &&  [ -f $HOME/.ssh/id_rsa  ]); then 
       ssh-add $HOME/.ssh/id{_dsa,_rsa,entity}
    else
       echo identities not generated ... first invoke .... ssh--keygen passphrase 
    fi
    
    
    echo ===== listing identities of the agent
    ssh-add -l
}


ssh--setup(){

  if [ $(ssh--agent-ok) == "1" ]; then
      echo agent is responding and holds identities
  else
     echo agent is not responding, trying to start a new one
     ssh--agent-start
  fi
}






	   ## demo of running a multi argument command on a remote node 
ssh-x(){
	X=${1:-$TARGET_TAG}  
	shift
	echo ssh $X "bash -lc \"$*\""
	     ssh $X "bash -lc \"$*\""
}


ssh--putkeys(){
  for target in $*
  do
     ssh--putkey $target
  done
}

ssh--putkey(){
    X=${1:-$TARGET_TAG}
    ssh $X "mkdir .ssh"
    cat ~/.ssh/id_{d,r}sa.pub | ssh $X "cat - >> ~/.ssh/authorized_keys2"
	ssh $X "chmod 700 .ssh ; chmod 700 .ssh/authorized_keys*" 

   # NB the permissions setting is crucial, without this the passwordless
   # connection fails, with no error message ... the keys are just silently ignored
   #
   #
   #  to setup passwordless from source to target need :
   #
   #     create the keys on the source machine
   #         source> ssh--keygen passphrase
   #
   #     copy the public keys to the target machine
   #         source> ssh--putkey target
   #
   #      start the agent and give it the keys on the source machine,
   #      the passphrase used at key creation will be prompted for 
   #         source>  ssh--agent-start
   #        
   # 
   #     
}

ssh--config(){

[ -d $HOME/.ssh ] || mkdir $HOME/.ssh 


## old versions of SSH do not like ForwardX11Trusted
if [ "$NODE_TAG" == "H" ]; then
  c="#"
else
  c=""
fi    

cat << EOC > $HOME/.ssh/config
#
#   do not edit, this is sourced from $HOME/$ENV_BASE/base/ssh.bash
#
#
host N
    user blyth
    hostname pdsf.nersc.gov
    
    
host VT
    #
    user hmmm
    hostname hmmm.bnl.gov
    protocol 2
    # NB the 127.0.0.1 is the callback ip on the remote side of the tunnel 
    # this tunnels local trafic on 5901 to the remote 5901
    LocalForward 5901 127.0.0.1:5901     
    
    
host G3
    hostname g3pb.ath.cx
    protocol 2 
host BP
    hostname bpost.kek.jp
	protocol 2
host LX
    hostname lxplus.cern.ch
	protocol 2
host X
    hostname 140.112.101.48
	user exist
host G1 
    hostname 140.112.102.250
	ForwardX11 yes
	ForwardX11Trusted yes
host I
    user blyth
    hostname 140.112.101.199
	protocol 1,2
	ForwardX11 yes
host H1
    user blyth
	hostname 140.112.101.41
host H
    user blyth
    hostname 140.112.101.48
host X
    hostname 140.112.101.48
	user exist
host P 
    hostname 140.112.102.250
    user dayabaysoft 
	ForwardX11 yes
$c	ForwardX11Trusted yes
host T
    user blyth
    hostname tersk.slac.stanford.edu
	UserKnownHostsFile /dev/null
##
## avoid the annoying need to delete the entry from the known_hosts file
## by specifying said file as /dev/null
##	
host L
	hostname pal.nuu.edu.tw	
	#user mahuang
	user sblyth 
	ForwardX11 yes
$c	ForwardX11Trusted yes
EOC


if [ "X$NODE_TAG" != "X$SOURCE_TAG" ]; then
cat << EOS >> $HOME/.ssh/config
## allows calling home
host G
    hostname g4pb.ath.cx
	user blyth
EOS
fi


}
	
