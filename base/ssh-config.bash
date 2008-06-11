





ssh--config(){

[ -d $HOME/.ssh ] || mkdir $HOME/.ssh 

local msg="=== $FUNCNAME:"

## old versions of SSH do not like ForwardX11Trusted
if [ "$NODE_TAG" == "H" ]; then
  c="#"
else
  c=""
fi    

local cfg=$HOME/.ssh/config
local pfg=$cfg.prior

if [ -f "$cfg" ]; then
   echo $msg moving $cfg to $pfg
   mv -f $cfg $pfg
fi

cat << EOC > $HOME/.ssh/config
#
#   do not edit, this is sourced from $ENV_HOME/base/ssh-config.bash
#
#
# /home/aberdeen/Datafiles
host A
    user aberdeen
	hostname aberdeentunnel.dyndns.org


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
    
host B
    hostname gateway.phy.bnl.gov
	protocol 2
	
host C
    hostname 140.112.101.190
              	     
host G3R
    hostname g3pb.ath.cx
    protocol 2 
	
host G3
    hostname 10.0.1.103
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
$c	ForwardX11Trusted yes
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
host S
    hostname simon.phys.ntu.edu.tw
	user blyth    
    
EOS
fi

  if [ -f "$pfg" ] && [ -f "$cfg" ]; then
     echo $msg comparing config files
	 ls -l $pfg $cfg
	 diff $pfg $cfg
  fi 



}

#
#   http://www.linuxjournal.com/node/6602/print 
#   claims 
#  ssh -L 9110:mail.example.net:110 shell.example.net
#   is equivalent to
#  ssh forwardpop 
#  
#   with the below in the config:
#  
# host forwardpop
#     Hostname shell.example.net
#     LocalForward 9110 mail.example.net:110 
#
