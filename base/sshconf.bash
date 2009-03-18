
sshconf-usage(){
  cat << EOU

   sshconf-gen
       generate the .ssh/config from 


EOU


}

sshconf-env(){
   elocal-
}

sshconf-src(){   echo base/sshconf.bash ; }
sshconf-source(){ echo ${BASH_SOURCE:-$ENV_HOME/$(sshconf-src)} ; }
sshconf-vi(){     vi $(sshconf-source) ; }

sshconf-gen(){

[ -d $HOME/.ssh ] || mkdir $HOME/.ssh 

local msg="=== $FUNCNAME:"

 ## old versions of SSH do not like ForwardX11Trusted
 ##   /home/blyth/.ssh/config: line 57: Bad configuration option: ForwardX11Trusted
 local c
 case $NODE_TAG in 
   H|XT) c="#" ;;
      *) c=""  ;;
 esac     


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

host AR
    user root 
    hostname aberdeentunnel.dyndns.org
 
host V
    hostname valiant.phys.vt.edu
    user dmohapat

host N
    hostname belle7.nuu.edu.tw
    user blyth

 
host L
    user blyth
    hostname simon.phys.ntu.edu.tw
 
host N
    user blyth
    hostname pdsf.nersc.gov
    
# IHEP     
host I
    user blyth
    hostname lxslc05.ihep.ac.cn
    protocol 2 
 
#
#  open tunnel session with  
#       G> ssh -vvv -L 8080:192.168.37.177:22 I
#   then  connect to the internal IP thru the tunnel
#       G> ssh -vvv XT 
#
#   ... this works without having to setup passwordless 
#    BUT how to do this for web packets ??? so that requests from
#    Safari take the trip thru the tunnel and ask apache on the internal machine port 8080
# 
#
host XT
    user blyth
    hostname localhost
    protocol 2 
    port 8080
    
            
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

# cms01	
host C
    user blyth
    hostname 140.112.101.190
    protocol 2 

# cms02
host C2
    user blyth
    hostname 140.112.101.191
    protocol 2 


# cms01     
host S
    hostname 140.112.101.190
    protocol 2 
    user dayabayscp                                 

# testing sending to locked down account on grid1
host S2
    hostname 140.112.102.250
    protocol 2 
    user dayabayscp                                 
    ForwardX11 no
                                                               	     
	
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
    protocol 2 
    hostname 140.112.101.48
host X
    hostname 140.112.101.48
	user exist
host P 
    hostname 140.112.102.250
    user dayabaysoft
    protocol 2 
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


  [ -f $HOME/.ssh/local-config ] && cat $HOME/.ssh/local-config  >> $HOME/.ssh/config
  

  echo $msg restricting access with chmod ...  needed for newer ssh ??
  chmod go-rw $cfg

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
