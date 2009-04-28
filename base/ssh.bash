
ssh--src(){ echo base/ssh.bash ; }
ssh--source(){ echo $(env-home)/$(ssh--src) ; }
ssh--vi(){ vi $(ssh--source) ; }
ssh--env(){ elocal- ; }
ssh--(){   . $(ssh--source) && ssh--env $* ; }  ## non standard locatio for precursor 

ssh--usage(){


cat << EOU

    NB the permissions setting is crucial, without this the passwordless
    connection fails, with no error message ... the keys are just silently ignored
   
   
     to setup passwordless from source to target need :
   
        create the keys on the source machine
            source> ssh--keygen 
        remember to skill the agent if a prior one is running 
   
        copy the public keys to the target machine
            source> ssh--putkey target
   
         start the agent and give it the keys on the source machine,
         the passphrase used at key creation will be prompted for 
            source>  ssh--agent-start
           
     ssh--info
          dump agent pid etc..

     ssh--tunnel <tag:N> <port:8080>
          tunnel remote port onto local machine ...
          remember yiou will probably also need to edit 
          ~/e/proxy/socks.pac and reload it in 
          Firefox > Preferences > 

     ssh--lskey 
          list keys in local authorized_keys2
      
          NB the entries indicate nodes/accounts from which this one can be 
             accessed vi  ssh key ... should be kept to the minimum needed 

     ssh--tags : $(ssh--tags)
     ssh--rlskey 
         list keys in all the remote nodes

 
     ssh--rmkey <type> <name> <node>
          delete keys from local authorized_keys2
          things that fit into a perlre can be used ie
         
             ssh--rmkey ".*" ".*" "pal.nuu.edu.tw"
             ssh--rmkey "..." "blyth" "al14"            
             ssh--rmkey  ".*" "blyth" "C2" 




     ssh--delkey <tag> <path-to-key>
           delete remote pubkey entry in authorized_keys{,2}           

     ssh--haskey <tag> <path-to-key>
           remote grep of authorized_keys{,2} to see if pubkey is present 

     ssh--addkey <tag> <path-to-key>

        Usage example :
           cd /tmp  ; scp N:.ssh/id_rsa.pub id_rsa.pub   ## grab the key of the new node
           ssh--addkey H id_rsa.pub                   ## append it on the target 
           rm id_rsa.pub

        This is useful to extend access to a node that accepts login only via key 
        to a new node, via transferring the nodes key via a
        node that already has keyed access. 

     ssh--inikey <tag> <path-to-key>

         Like addkey but scrub prior authorized_keys{,2} entries 





     ssh--key2base <path-to-key>
     ssh--key2tag  <path-to-key>
          extract the tag and base when the key name conforms to convention 
          eg for name P.id_rsa ... would return base id_rsa and tag P

      ssh--designated-key : $(ssh--designated-key) 
      ssh--distribute-key  <path-to-key> tag1 tag2 ...
          distribute the public key into the authorized_keys2 on the 
          destination tags 
          NB no checking for duplicates 



  







    ssh--appendtag <target-tag> <new-tag> 
        NOT YET IMPLEMENTED
        
        Automate the key grabbing and cleanup of the above example allowing : 
            ssh--appendtag H N


    Related function precursors ..
        sshconf-     ## used for generating the .ssh/config file



EOU

}



ssh--tunnel(){
  local tag=${1:-N} 
  local port=${2:-8080}

  local cmd="ssh -fND localhost:$port $tag "

  cat << EON

     -D Specifies a local ``dynamic'' application-level port forwarding.  This works by allocating a socket to listen to port on the local side, optionally bound to the specified bind_address. 
     -N no remote command, just forward
     -f go to background 
  

   kill the process to stop the tunnel 

EON

 echo $msg opening ...  $cmd 
 eval $cmd 
 sleep 1 
 ps aux | grep ssh

}

ssh--mvkey-(){
  local path=$1
  local stamp=$(base-pathstamp $path)
  cat << EOC
mv $path          $path.$stamp 
mv $path.pub  $path.pub.$stamp
EOC
}

ssh--keygen(){
  local msg="=== $FUNCNAME :"
  local passph
  read -s -p "$msg Enter passphrase:" passph
 
  local passph2
  read -s -p "$msg Enter passphrase again to confirm :" passph2

  [ "$passph" != "$passph2" ] && echo $msg ABORT the passphrases do not match .. && return 1


  [ "$passph" == "dummy" -o "$passph" == "" ] && echo "you must enter a non blank passphrase " && return 
  [ -d "$HOME/.ssh" ] || ( mkdir $HOME/.ssh && chmod 700 $HOME/.ssh )

  echo generating keys on node $NODE_TAG
  local types="rsa dsa"
  for typ in $types
  do	  
     case $typ in
           rsa1) keyname=identity ;;
	   rsa)  keyname=id_rsa   ;;
	   dsa)  keyname=id_dsa   ;;
	     *)  keyname=error    ;;
     esac    
     keyfile="$HOME/.ssh/$keyname"
     if [ -f "$keyfile" ]; then
	 echo "$msg keyfile $keyfile already exists ...  " 
         ssh--mvkey- $keyfile         
         local ans
         read -p "$msg proceed to move them aside ? enter YES to proceed " ans
         [ "$ans" != "YES" ] && $msg aborting && return 1
         local mmd
         ssh--mvkey- $keyfile | while read mmd ; do
            echo $mmd
            eval $mmd
         done
     fi
     ssh-keygen -t $typ -f $keyfile  -C "$USER@$NODE_TAG "  -N $passph 
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
    
    #if ([ -f $HOME/.ssh/identity ] && [ -f $HOME/.ssh/id_dsa ] &&  [ -f $HOME/.ssh/id_rsa  ]); then 
    #   ssh-add $HOME/.ssh/id{_dsa,_rsa,entity}
    #else
    #   echo identities not generated ... first invoke .... ssh--keygen passphrase 
    #fi
    
	 if ([ -f $HOME/.ssh/id_dsa ] &&  [ -f $HOME/.ssh/id_rsa  ]); then 
       ssh-add $HOME/.ssh/id{_dsa,_rsa}
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
  local tags="${1:-$BACKUP_TAG}"
  for target in $tags
  do
     ssh--putkey $target
  done
}

ssh--putkey(){
    X=${1:-$TARGET_TAG}
    ssh $X "mkdir .ssh"
    cat ~/.ssh/id_{d,r}sa.pub | ssh $X "cat - >> ~/.ssh/authorized_keys2"
    ssh $X "chmod 700 .ssh ; chmod 700 .ssh/authorized_keys*" 

}




ssh--key2ak(){
   local name=$(basename $1)
   case $name in 
          *.id_rsa.pub) echo authorized_keys2 ;;
          *.id_dsa.pub) echo authorized_keys2 ;;
            id_rsa.pub) echo authorized_keys2 ;;
            id_dsa.pub) echo authorized_keys2 ;;
          identity.pub) echo authorized_keys  ;;
        *.identity.pub) echo authorized_keys  ;;
                     *) echo ERROR ;;
   esac
}

ssh--addkey(){
   local msg="=== $FUNCNAME :"
   local tag=${1:-$TARGET_TAG}
   local key=${2:-$(ssh--local-key)}
   ! ssh--oktag- && echo $msg skipping excluded tag $tag && return 1 
   [ ! -f "$key" ] && echo $msg ABORT key $key does not exist && return 1
   local ak=$(ssh--key2ak $key)
   [ "$ak" == "ERROR" ] && echo $msg ABORT key name of $key is not supported && return 2
   [ "$(ssh--haskey $tag $key)" == "YES" ] && echo $msg tag $tag already has key $key ... skipping  && return 0

   cat $key | ssh $tag "cat - >> ~/.ssh/$ak"              
}

ssh--haskey(){
   local tag=$1
   local key=$2
   local ak=$(ssh--key2ak $key)
   [ "$ak" == "ERROR" ] && return 1

   cat $key | ssh $tag "grep \"$(cat -)\" ~/.ssh/$ak > /dev/null && echo YES || echo NO"  2> /dev/null
}

ssh--delkey(){
   local msg="=== $FUNCNAME :"
   local tag=$1
   local key=$2
   ! ssh--oktag- && echo $msg skipping excluded tag $tag && return 1 
   local ak=$(ssh--key2ak $key)
   [ "$ak" == "ERROR" ] && echo $msg ABORT key name of $key is not supported && return 2
   [ "$(ssh--haskey $tag $key)" == "NO" ] && echo $msg tag $tag does not have key $key ... skipping deletion  && return 0

   cat $key | ssh $tag "cd .ssh && cp $ak $ak.tmp && grep -v \"$(cat -)\" $ak.tmp > $ak && rm $ak.tmp  "
}

ssh--oktag-(){
   local tag=$1
   case $tag in
      H|G) return 1  ;;
   esac
}

ssh--inikey(){
   local msg="=== $FUNCNAME :"
   local tag=${1:-$TARGET_TAG}
   local key=${2:-$(ssh--local-key)}
   ! ssh--oktag- && echo $msg skipping excluded tag $tag && return 1 
   [ ! -f "$key" ] && echo $msg ABORT key $key does not exist && return 1
   local ak=$(ssh--key2ak $key)
   [ "$ak" == "ERROR" ] && echo $msg ABORT key name of $key is not supported && return 2

   cat $key | ssh $tag "mkdir -p .ssh ; chmod 700 .ssh ; cat - > .ssh/$ak ; chmod 600 .ssh/$ak "              
}



ssh--lskey(){
     cat $HOME/.ssh/authorized_keys2 | perl -n -e 's,^ssh-(\S*) (\S*) (.*)$, $1 $3  , && print ' -  
}


ssh--tags(){
   cat << EOT
H1 
P 
C 
N 
G1
H
EOT
  
}

ssh--rlskey(){
   ## need tags for root ???
   echo $msg list keys on remote nodes : $tags
   local tag
   for tag in $(ssh--tags) ; do 
      echo ""
      ssh--rlskey- $tag
   done

}

ssh--rlskey-(){
  local tag=${1:-P}
  local msg="=== $FUNCNAME :"
  echo $msg $tag  
  ssh $tag "bash -lc \"  ssh-;ssh--lskey; ls -alst ~/.ssh/     \""
}


ssh--local-key(){      echo $HOME/.ssh/id_rsa.pub ; }
ssh--designated-key(){ echo $HOME/.ssh/$(env-designated).id_rsa.pub ; }
ssh--hub-key(){        echo $HOME/.ssh/id_rsa.pub ; }


ssh--grab-key(){
   local msg="=== $FUNCNAME :"
   
   local path=${1:-$(ssh--designated-key)}
   local tag=$(ssh--key2tag $path)
   local base=$(ssh--key2base $path)

   echo $msg path $path tag $tag base $base 
   [ -f $path    ] && echo $msg $path is already present && return 0
   [ ! -f $path ]  && scp $tag:~/.ssh/$base $path  
   [ ! -f $path ]  && echo $msg FAILED to grab $path from $tag ... you need to ssh--keygen on $tag  && return 1
}



ssh--key2tag(){
   local msg="=== $FUNCNAME :"
   local path=${1:-$(ssh--designated-key)}
   local name=$(basename $path)
   local dtag=${name/.*/}
   echo $dtag 
}

ssh--key2base(){
   local msg="=== $FUNCNAME :"
   local path=${1:-$(ssh--designated-key)}
   local name=$(basename $path)
   local dtag=$(ssh--key $path)
   local n=$(( ${#dtag} + 1 ))
   local base=${name:$n}
   echo $base
}

ssh--ishub-(){ [ "$NODE_TAG" == "$(ssh--hubtag)" ] && return 0 || return 1 ; }
ssh--hubtag(){  echo G ; }

ssh--distribute-key(){

   local msg="=== $FUNCNAME :"
   ! ssh--ishub- && echo $msg ABORT this must be run from hub node $(ssh--hubtag) && return 1 

   local path=${1:-$(ssh--designated-key)}
   shift
   local tags=$*

   local ans
   read -p "$msg $path to nodes : $tags , enter YES to proceed " ans
   [ "$ans" != "YES" ]    && echo $msg skipping && return 1

   local tag
   for tag in $tags ; do
      echo $msg ... $tag $path
      case ${SSH__DISTRIBUTE_KEY:-add} in 
          INI) ssh--inikey  $tag $path ;;
          add) ssh--addkey  $tag $path ;;
      esac
   done
}


ssh--initialize-authkeys(){

   local msg="=== $FUNCNAME :"
   ! ssh--ishub- && echo $msg ABORT this must be run from hub node $(ssh--hubtag) && return 1 
   local hubkey=$(ssh--hub-key)
   local tags=$(ssh--tags)

   cat << EOM
$msg CAUTION : this scrubs ALL authkeys on ALL NODES : $tags 
$msg and then distributes the hub public key $hubkey to them , 

THIS RISKS LOCKOUT FROM KEY ONLY NODES ... LIKE : H  
HAVE AN ACTIVE CONNECTION ON THESE NODES BEFORE RUNNING THIS 
AND TEST CONNECTIVITY  AFTER DOING SO


EOM

   local ans
   read -p "$msg enter YES to proceed "  ans
   [ "$ans" != "YES" ] && echo $msg skipping && return 0

   SSH__DISTRIBUTE_KEY=INI ssh--distribute-key $hubkey $tags 
}


ssh--server-authkeys(){

   local msg="=== $FUNCNAME :"
   ! ssh--ishub- && echo $msg ABORT this must be run from hub node $(ssh--hubtag) && return 1 

   echo $msg distribute the server public key to its backup nodes

   local serverkey=$(ssh--designated-key)
   ssh--grab-key $serverkey                   || return 1 
   local dtag=$(ssh--key2tag $serverkey)
   local tags=$(local-backup-tag $dtag)

   ssh--distribute-key $serverkey $tags       || return 1

   ## need designation of scponly endpoints in order that 
   ## requisite keys are in the right place  

}

ssh--selectkey(){
    local type=${1:-dss}
    local name=${2:-sblyth}
    local node=${3:-pal.nuu.edu.tw}
    echo $type
    perl -n -e "s,^ssh-($type).*($name\@$node)\$,\$1 \$2, && print " $(ssh--ak2)
}

ssh--ak2(){ echo $HOME/.ssh/authorized_keys2 ; }
ssh--kvi(){ vi $(ssh--ak2) ; }
ssh--rmkey(){
    local msg="=== $FUNCNAME :"
    local tmp=/tmp/env/$FUNCNAME && mkdir -p $tmp
    local type=${1:-dss}
    local name=${2:-sblyth}
    local node=${3:-pal.nuu.edu.tw}
    local sak=$(ssh--ak2)
    local tak=$tmp/$(basename $sak)
    perl -p -e "s,^ssh-($type).*($name\@$node)\n,,s" $sak >  $tak
    cat << EOM
$msg  
  type : $type  
  name : $name 
  node : $node
EOM
    local cmd="diff $sak $tak"
    echo $msg $cmd
    eval $cmd
    local ans
    read -p "$msg proceed with this key removal ? YES to proceed " ans
    [ "$ans" != "YES" ] && echo $msg skipping && return 0

    cp $tak $sak
    chmod 600 $sak
    rm -rf $tmp
}

ssh--createdir(){

   local msg="=== $FUNCNAME :"
   local home=$1
   local user=$(basename $home)
   [ "$home" == "$HOME" ] && echo $msg THIS ONLY WORKS FOR OTHER USERS ... NOT YOURSELF && return 1

   local dir=$home/.ssh
   sudo -u $user mkdir $dir
   sudo -u $user chmod 700 $dir 
}

ssh--cmd(){
  local tag=$1
  shift 
  ssh $tag "bash -c \" $*  \" > /dev/null && echo YES || echo NO  "  2> /dev/null	
}


