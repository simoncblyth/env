sshconf-src(){   echo base/sshconf.bash ; }
sshconf-source(){ echo ${BASH_SOURCE:-$ENV_HOME/$(sshconf-src)} ; }
sshconf-vi(){     vi $(sshconf-source) ; }

sshconf-usage(){
  cat << EOU

    sshconf-gen
          generate the .ssh/config 

    sshconf-gen- <user:$USER>
          emit the config for relevant tags for the user 

    sshconf-tag- <tag>
          emit the sshconf for a single tag 

    
   Most of the details are now sourced in : $(local-source) 
   edit with local-vi
    

EOU

}

sshconf-env(){
   elocal-
}

sshconf-tag-(){
   local tag=$1
   local user=$(local-tag2user $tag)

   cat << EOC

# $(local-tag2node $tag)
host $tag
    user $user
    hostname $(local-tag2ip $tag)
    protocol 2
EOC
   
   ## old versions of SSH do not like ForwardX11Trusted
   local c
   case $NODE_TAG in 
      H|XT) c="#" ;;
         *) c=""  ;;
   esac     

   case $tag in
      G1|L|P) echo "    ForwardX11 yes" ;; 
    S|SC2|S2) echo "    ForwardX11 no" ;; 
   esac
   case $tag in
      G1|L|P) echo "$c   ForwardX11Trusted yes" ;;
   esac
   case $tag in
      T|LX) echo "    UserKnownHostsFile /dev/null" ;;
   esac 
  ## avoids the annoying need to delete the entry from the known_hosts file by as /dev/null

}



sshconf-gen-(){
   local msg="=== $FUNCNAME :"
   local tags=$(local-tags $*)

   cat << EOH
# $msg $* tags : $tags   
EOH
   local tag
   for tag in $(local-tags $*) ; do
      sshconf-tag- $tag
   done
}

sshconf-gen(){

   local msg="=== $FUNCNAME:"
   [ -d $HOME/.ssh ] || mkdir $HOME/.ssh 
   local cfg=$HOME/.ssh/config
   local tmp=/tmp/env/$FUNCNAME && mkdir -p $tmp
   local tfg=$tmp/config

   sshconf-gen- > $tfg 
   [ -f $HOME/.ssh/local-config ] && cat $HOME/.ssh/local-config  >> $tfg
   [ -f $HOME/.ssh-local-config ] && cat $HOME/.ssh-local-config  >> $tfg
  
   echo $msg restricting access with chmod ...  
   chmod go-rw $cfg

   if [ -f "$tfg" ] && [ -f "$cfg" ]; then
         echo $msg comparing config files
	 ls -l $cfg $tfg
         echo diff $cfg $tfg
         diff $cfg $tfg
   else
         echo $msg created new candidate for $cfg in $tfg
   fi 
   local ans
   read -p "$msg do you want to accept these config changes ? enter YES to proceed " ans
   [ "$ans" != "YES" ] && echo $msg skipping && return 0
 
   cp $tfg $cfg 
   rm $tfg

}


