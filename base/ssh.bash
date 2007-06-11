
#
#   machines that are currently resisting arrest:
#       pal@nuu "L"
#

SSH_BASE=".ssh"


ssh--keygen(){

  passph=${1:-dummy}
  [ "$passph" == "dummy" ] && echo "you must enter a passphrase as the argument " && return 
  [ -d "$HOME/$SSH_BASE" ] || ( mkdir $HOME/$SSH_BASE && chmod 700 $HOME/$SSH_BASE )

  echo generating keys on node $NODE_TAG
  for typ in "rsa1 rsa dsa"
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
          ssh-keygen -t $typ -f $keyfile  -C "$typ from $NODE_TAG "  -N $passph   
	  fi	  
  done	
	
}



	   ## demo of running a multi argument command on a remote node 
ssh-x(){
	X=${1:-$TARGET_TAG}  
	shift
	echo ssh $X "bash -lc \"$*\""
	     ssh $X "bash -lc \"$*\""
}


ssh-putkey(){
    X=${1:-$TARGET_TAG}
    cat ~/.ssh/id_{d,r}sa.pub | ssh $X "cat - >> ~/.ssh/authorized_keys2"
	ssh $X "chmod 700 .ssh ; chmod 700 .ssh/authorized_keys*" 

   ## NB the permissions setting is crucial, without this the passwordless
   ## connection fails, with no error message ... the keys are just silently ignored
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
host G1 
    user blyth
    hostname 140.112.102.250
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
	
