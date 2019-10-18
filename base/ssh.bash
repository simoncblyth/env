
ssh--src(){ echo $BASH_SOURCE ; }
ssh-vi(){  vi $BASH_SOURCE ; }
ssh--vi(){ vi $BASH_SOURCE ; }
ssh--env(){ elocal- ; }
#ssh--(){   . $(ssh--source) && ssh--env $* ; }  ## non standard locatio for precursor 


ssh--usage(){ cat << EOU

*ssh--* Bash Functions
==========================

.. contents:: :local:

References
------------

* debugging tips http://blog.codefront.net/2007/02/28/debugging-ssh-public-key-authentication-problems/

Passwordless SSH setup
-----------------------------

.. warning:: the permissions setting is crucial, without this the passwordless connection fails, with no error message the keys are just silently ignored

Create keys on source machine
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Remember to kill the agent if a prior one is running:: 

    source> ssh--keygen 
     
A passphrase will be promted for twice, and key pairs created at::

    ~/.ssh/id_rsa
    ~/.ssh/id_dsa
    ~/.ssh/id_rsa.pub
    ~/.ssh/id_dsa.pub


Copy public keys to target
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you have password access to the target::

    source> ssh--putkey target
  
Otherwise copy/paste or attach the public *.pub* keys 
to an email and send to administrator of target machine.

Start the SSH agent on source machine
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Start the agent and give it the keys on the source machine,
the passphrase used at key creation will be prompted for 

::

    source>  ssh--agent-start



SSH Key Juggling
-----------------

As direct ssh connection from D to N is blocked (by network gnomes) are 
making such connections via C, using a forced ssh command that only 
kicks in when an alternative key is used::

     61 # belle7 via forced command on C
     62 host CN
     63     user blyth
     64     hostname 140.112.101.190
     65     IdentityFile ~/.ssh/alt_dsa
     66     protocol 2

Problem is that when the normal ssh agent is running providing 
passwordless access to all non-blocked nodes this uses the 
normal key and thus doesnt run the forced command to get to N, 
ending up at C::

    delta:~ blyth$ ssh CN
    Scientific Linux CERN SLC release 4.8 (Beryllium)
    Last login: Thu Dec 11 12:04:14 2014 from simon.phys.ntu.edu.tw
    [blyth@cms01 ~]$ 


Workaround (not solution as need to manually enter password)
is to prevent interference from the agent when connecting to CN::

    delta:env blyth$ SSH_AUTH_SOCK= ssh CN
    Scientific Linux CERN SLC release 4.8 (Beryllium)
    Enter passphrase for key '/Users/blyth/.ssh/alt_dsa': 
    Last login: Thu Dec 11 12:04:27 2014 from cms01.phys.ntu.edu.tw
    [blyth@belle7 ~]$ 

Or equivalently with function::

    delta:env blyth$ CN(){ SSH_AUTH_SOCK= ssh CN ; }
    delta:env blyth$ CN
    Scientific Linux CERN SLC release 4.8 (Beryllium)
    Enter passphrase for key '/Users/blyth/.ssh/alt_dsa': 
    Last login: Thu Dec 11 12:12:00 2014 from cms01.phys.ntu.edu.tw
    [blyth@belle7 ~]$ 




Debugging Tips
----------------

Connect without the keys
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For debugging or to avoid running a "forced" command (eg from gateway to internal node)
it is sometimes useful to revert to password authentication. Do so with::

     source> ssh -o PubkeyAuthentication=no VT

This setting can be persisted in the config file under a new host name
(add to ~/.ssh-local-config then *sshconf-gen*  if using *sshconf-*)
::

     host VTN
          hostname gateway.domain
          protocol 2
          PubkeyAuthentication no


SSH from cron issues
~~~~~~~~~~~~~~~~~~~~~~~

Passwordless ssh requires the ssh-agent to be running and authenticates
and some envvars to identify the agent.

::

    [blyth@belle7 ~]$ env -i SSH_AUTH_SOCK=/tmp/ssh-pXBvj24135/agent.24135 SSH_AGENT_PID=24136 ssh N1 hostname
    belle1.nuu.edu.tw
    [blyth@belle7 ~]$ env -i SSH_AUTH_SOCK=/tmp/ssh-pXBvj24135/agent.24135 ssh N1 hostname
    belle1.nuu.edu.tw
    [blyth@belle7 ~]$ env -i ssh N1 hostname
    Enter passphrase for key '/home/blyth/.ssh/id_dsa': 

    [blyth@belle7 ~]$ cat .ssh-agent-info-N 
    SSH_AUTH_SOCK=/tmp/ssh-pXBvj24135/agent.24135; export SSH_AUTH_SOCK;
    SSH_AGENT_PID=24136; export SSH_AGENT_PID;
    #echo Agent pid 24136;


Troubleshoot Passwordless access not working
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ssh works via client-daemon with config files for 
both client (man ssh_config) and daemon (man sshd_config) 
on both the nodes you are connecting 

All 4 config files need to be reviewed and have:

* client settings should use "Protocol 2" or "2,1" (NOT with 1 first)
* daemon settings should use *AuthorizedKeysFile .ssh/authorized_keys2*


Recover from a forgotten passphrase 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#. create a new key pair 
#. transfer public keys to target and append to authorized_keys2

::

   scp ~/Downloads/id_dsa.pub C:.ssh/dybdb1.id_dsa.pub
   scp ~/Downloads/id_rsa.pub C:.ssh/dybdb1.id_rsa.pub
   C >  cd .ssh ; cat dybdb1.id_dsa.pub dybdb1.id_rsa.pub >> authorized_keys2

NB this can be automated with *ssh--putkey* when you have control of both ends 

Appending keys for restricted no-login shell identity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    sudo vi  ~dayabayscp/.ssh/authorized_keys2
    sudo bash -c "cat dybdb1.* >>  ~dayabayscp/.ssh/authorized_keys2"  



Access remote node via gateway
--------------------------------

* http://www.onlamp.com/pub/a/onlamp/excerpt/ssh_11/index1.html

The problem: network gnomes have blocked directed access `D -> N`
but access `D -> C` and `C -> N` remains operational.

To avoid an extra manual ssh step and prevent direct running 
of commands over ssh : need to setup a forced command 
on the gateway.

On originating node D, generate an alternative key::

    delta:.ssh blyth$ ssh-keygen -t rsa -f alt_rsa -C "alt $USER@$NODE_TAG"
    Generating public/private rsa key pair.
    Enter passphrase (empty for no passphrase): 
    Enter same passphrase again: 
    Your identification has been saved in alt_rsa.
    Your public key has been saved in alt_rsa.pub.

    delta:.ssh blyth$ ssh-keygen -t dsa -f alt_dsa -C "alt $USER@$NODE_TAG"
    Generating public/private dsa key pair.
    Enter passphrase (empty for no passphrase): 
    Enter same passphrase again: 
    Your identification has been saved in alt_dsa.
    Your public key has been saved in alt_dsa.pub.
    ...


Add the key to the agent::

    delta:.ssh blyth$ ssh-add -l
    1024 4d:f7:32:71:1e:09:6a:2b:02:1b:91:5c:49:fd:1c:04 /Users/blyth/.ssh/id_dsa (DSA)
    2048 2c:3a:e2:f7:e4:04:7e:62:08:fd:dc:7b:19:ec:24:10 /Users/blyth/.ssh/id_rsa (RSA)

    delta:.ssh blyth$ ssh-add alt_rsa
    Enter passphrase for alt_rsa: 
    Identity added: alt_rsa (alt_rsa)

    delta:.ssh blyth$ ssh-add -l
    1024 4d:f7:32:71:1e:09:6a:2b:02:1b:91:5c:49:fd:1c:04 /Users/blyth/.ssh/id_dsa (DSA)
    2048 2c:3a:e2:f7:e4:04:7e:62:08:fd:dc:7b:19:ec:24:10 /Users/blyth/.ssh/id_rsa (RSA)
    2048 b0:9c:a6:c9:f5:80:97:65:48:a3:76:fb:04:62:2b:37 alt_rsa (RSA)


Copy public key to gateway::

    delta:.ssh blyth$ scp alt_rsa.pub C:.ssh/ 
    delta:.ssh blyth$ scp alt_dsa.pub C:.ssh/ 

On the gateway machine::

    [blyth@cms01 .ssh]$ cat alt_rsa.pub >> authorized_keys2
    [blyth@cms01 .ssh]$ cat alt_dsa.pub >> authorized_keys2

Edit the authorized_keys2 adding a forced command infront of the 
public key just added:: 

    [blyth@cms01 .ssh]$ vi authorized_keys2

    command="sh -c 'ssh N ${SSH_ORIGINAL_COMMAND:-}'" ssh-rsa AAAAB3NzaC1y
        ## this works, but promts for password for the key of gateway machine

    command="sh -c 'source ~/.ssh-agent-info ; ssh N ${SSH_ORIGINAL_COMMAND:-}'" ssh-dss AAAAB
        ## sourcing the agent info in the forced command, allows to make the two hops passwordless-ly

Back on original machine add .ssh/config entry that uses the alternative key, 
and *ssh-add* that to the agent::

    host CN
        user blyth
        hostname 140.112.101.190
        IdentityFile ~/.ssh/alt_rsa
        protocol 2

On gateway, copy the alt key forward to target and append to authorized_keys2 on target::

    [blyth@cms01 .ssh]$ scp alt_dsa.pub N:.ssh/
    [blyth@belle7 .ssh]$ cat alt_dsa.pub >> authorized_keys2


**Success** : tis fiddly to get working due to too many moving parts,
but can now hop around network blockages and run remote commands::

    delta:~ blyth$ ssh CN
    Scientific Linux CERN SLC release 4.8 (Beryllium)
    Last login: Thu Oct 23 14:38:13 2014 from cms01.phys.ntu.edu.tw
    [blyth@belle7 ~]$ 

    delta:~ blyth$ ssh CN hostname
    Scientific Linux CERN SLC release 4.8 (Beryllium)
    belle7.nuu.edu.tw




Private web server access over SSH 
------------------------------------

The problem: a remote node K is running a web server that you wish to 
test from your laptop but the remote node is not web accessible. 
How can you conveniently check webserver responses 
from browsers or commandlines on your laptop ?

Fork a tunnel
~~~~~~~~~~~~~~

Start the tunnel on laptop. The process goes to background so the terminal session can
be closed without stopping the tunnel process::

    simon:~ blyth$ ssh--
    simon:~ blyth$ ssh--tunnel K 9090

         -D Specifies a local dynamic'' application-level port forwarding.  
            This works by allocating a socket to listen to port on the local side, 
            optionally bound to the specified bind_address. 
         -N no remote command, just forward
         -f go to background 

       kill the process to stop the tunnel 

    opening tunnel with command ...  ssh -fND localhost:9090 K

Alternatively without the *ssh--tunnel* bash function can just directly do::

    ssh -fND localhost:9090 K

Check the tunnel process::

    simon:e blyth$ ps aux | grep localhost:9090
    blyth    20464   0.0  0.0    77864    488   ??  Ss   12:43pm   0:00.01 ssh -fND localhost:9090 K


.. warning:: This ties up localhost:9090 so attempting to connect a local server instance on port 9090 at the same time as the tunnel is running fails 


Direct access does not work
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Because the request is not going via the local 9090 port setup::

    simon:~ blyth$ curl http://130.87.106.59:9090/servlet/db/
    ^C

Socks aware commandline clients like curl
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Some commandline tools support routing requests via the SOCKS proxy::

    simon:~ blyth$ curl --socks5 localhost:9090 http://130.87.106.59:9090/servlet/db/
    <exist:result xmlns:exist="http://exist.sourceforge.net/NS/exist">
        <exist:collection name="/db" owner="admin" group="dba" permissions="rwurwurwu">
            <exist:collection name="test" created="Jun 20, 2013 08:56:20" owner="guest" group="guest" permissions="rwur-ur-u"/>
            <exist:collection name="hfagc_tags" created="Jun 20, 2013 08:56:20" owner="admin" group="dba" permissions="rwur-ur-u"/>
            <exist:collection name="hfagc" created="Jun 20, 2013 08:56:14" owner="admin" group="dba" permissions="rwur-ur-u"/>
        </exist:collection>
    </exist:result>


That is equivalent to ssh-ing into the remote node and running the query locally::

    b2mc:~ heprez$ curl http://130.87.106.59:9090/servlet/db/
    <exist:result xmlns:exist="http://exist.sourceforge.net/NS/exist">
        <exist:collection name="/db" owner="admin" group="dba" permissions="rwurwurwu">
            <exist:collection name="test" created="Jun 20, 2013 08:56:20" owner="guest" group="guest" permissions="rwur-ur-u"/>
            <exist:collection name="hfagc_tags" created="Jun 20, 2013 08:56:20" owner="admin" group="dba" permissions="rwur-ur-u"/>
            <exist:collection name="hfagc" created="Jun 20, 2013 08:56:14" owner="admin" group="dba" permissions="rwur-ur-u"/>
        </exist:collection>
    </exist:result>


Configure OSX Safari to use the SOCKS proxy 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://apple.stackexchange.com/questions/168474/how-to-make-apple-mail-app-to-use-socks-proxy-configured-by-a-pac-file


Add an *if* section to the *~/env/proxy/socks.pac* file  
corresponding to the URL want to check::

   function FindProxyForURL(url, host) {
   ...
   if (url.substring(0,25) ==   "http://130.87.106.59:9090" ){
      return "SOCKS 127.0.0.1:9090" ;
   }  
   ...
   return "DIRECT" ;
   }


.. warning:: experience suggests it is more reliable to use IP addresses rather than names 


#. go to *System Preferences/Network/Advanced.../Ethernet/Proxies/Configure Proxies:Using a PAC file*
#. select the file *~/env/proxy/socks.pac*
#. Then click *APPLY* button and close the window with top left red icon
#. exit Safari.app if already running and open it again 

* :env:`/wiki/Tunneling`

#. test by enterinng URLs into Safari URL bar like, 

   * http://130.87.106.59:9090/servlet/db/
   * http://130.87.106.59:9090/xmldb/db/


NB this will stop working when the session establishing the tunnel is closed 


Resuming a broken tunnel
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Check the tunnel process::

    simon:e blyth$ ps aux | grep localhost:9090
    blyth    20464   0.0  0.0    77864    488   ??  Ss   12:43pm   0:00.01 ssh -fND localhost:9090 K


After killing the tunnel process, curl responds with *connection fails* almost instantaneously::

    simon:e blyth$ kill -9 20464 
    simon:e blyth$ curl --socks5 localhost:9090 http://130.87.106.59:9090/servlet/db/
    curl: (7) couldnt connect to host
    simon:proxy blyth$ 

Similarly Safari says "Contacting 130.87.106.59" and after a minute or so gives an error page::

    Safari cant open the page.
    Safari cant open the page http://130.87.106.59:9090/xmldb/db/test/ because the server where this page is located isnt responding.


Starting another tunnel immediately gives curl and Safari access again.




How to handle scponly nodes + roots keys ? 
----------------------------------------------

Receive keys as copy/pastes within email on Mac laptop. Paste em into files and scp over to target::

    simon:~ blyth$ pbpaste > D2.id_rsa.pub
    simon:~ blyth$ pbpaste > D2.id_dsa.pub
    simon:~ blyth$ scp D2* C:

    [blyth@cms01 ~]$ sudo bash -c "cat D2* >> ~dayabayscp/.ssh/authorized_keys2 "
    [blyth@cms01 ~]$ sudo vi  /home/dayabayscp/.ssh/authorized_keys2    # manually added some new lines



Utilities and informational functions
----------------------------------------

*ssh--info*
          dump agent pid etc..

*ssh--tunnel <tag:N> <port:8080>*
          tunnel remote port onto local machine ...
          remember you will probably also need to edit 
          *~/e/proxy/socks.pac* and reload it in 
          Firefox > Preferences > 

          An issue with this is that privileged ports can only be
          forwarded by root, but the nodes that would want to tunnel
          to usually would have password access switched off so that means
          would have to setup ssh keys for root.

          So probably easier to prick holes in the iptables for specific ips
          while testing 

*ssh--lskey* 
          list keys in local authorized_keys2
      
          the entries indicate nodes/accounts from which this one can be 
          accessed via ssh key. These should be kept to the minimum needed 

*ssh--tags* 
          list of remote nodes that are ssh accessible
     
*ssh--rlskey* 
          list keys in all the remote nodes


Server/Backup Management
--------------------------

.. warning:: not yet operational

When using a hub node which is backed up to 
multiple backup nodes, there can be quite a few keys to juggle.

         
*ssh--designated-key*
           the pubkey for the node that is currently the designated server

*ssh--designated-tags* 
           tags of the backup nodes for the designated server 

*ssh--server-authkeys*
           Grabs the designated key, is not already present and distributes 
           it to the backup nodes that need it. 

           This needs to be rerun 
           after changing backup tags or designated server
           OR can be re-run just to check the keys  are inplace 
              

Deprecated early incarnation 
-----------------------------

*ssh--rmkey <type> <name> <node>*

          delete keys from local authorized_keys2
          things that fit into a perlre can be used ie::
         
             ssh--rmkey ".*" ".*" "pal.nuu.edu.tw"
             ssh--rmkey "..." "blyth" "al14"            
             ssh--rmkey  ".*" "blyth" "C2" 


Basis functions for key management 
------------------------------------

*ssh--delkey <tag> <path-to-key>*
        delete remote pubkey entry in authorized_keys{,2}

*ssh--haskey <tag> <path-to-key>*
        remote grep of authorized_keys{,2} to see if pubkey is present 

*ssh--addkey <tag> <path-to-key>*
        This is useful to extend access to a node that accepts login only via key
        to a new node, via transferring the nodes key via a
        node that already has keyed access.::

            cd /tmp  ; scp N:.ssh/id_rsa.pub id_rsa.pub   ## grab the key of the new node
            ssh--addkey H id_rsa.pub                      ## append it on the target 
            rm id_rsa.pub


*ssh--inikey <tag> <path-to-key>*
        Like addkey but scrub prior authorized_keys{,2} entries

*ssh--distribute-key  <path-to-key> tag1 tag2 etc*
        distribute the public key into the authorized_keys2 on the destination tags

*ssh--retract-key  <path-to-key> tag1 tag2 etc*
        delete the public key from the authorized keys of the destination tags


Functions depending/supporting a key naming convention 
---------------------------------------------------------

*ssh--designated-key*
          path to designated key 

*ssh--key2base <path-to-key>*
          extracted base eg **id_rsa** assuming key naming convention  

*ssh--key2tag  <path-to-key>*
          extracted tag eg **P** for name **P.id_rsa**

*ssh--grab-key <local-path-to-key>*
          Copy the key from remote node and store at the specified path, the 
          form of the basename must follow the naming convention in order to identify 
          which node to get it from and which type of key to get. eg::

                ssh--grab-key $HOME/.ssh/YY.id_rsa.pub   # grab remote key YY:.ssh/id_rsa.pub

Issues
-------

openssh/openssl version mismatch 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* http://dayabay.phys.ntu.edu.tw/tracs/env/ticket/328

resolved by uninstall/reinstall : but git was a casualty

::

    simon:~ blyth$ sudo port uninstall openssh @5.5p1_2
    --->  Unable to uninstall openssh @5.5p1_2, the following ports depend on it:
    --->  git-core @1.6.3.1_0+doc+svn
    --->  git-core @1.7.2.2_0+doc
    Error: port uninstall failed: Please uninstall the ports that depend on openssh first.
    simon:~ blyth$



SSH Hardening suggestions
----------------------------

In /etc/sshd_config you might want to uncomment the lines::

     PasswordAuthentication no
     PermitEmptyPasswords no

Also, you want to make sure you have::
   
     RSAAuthentication yes
     PubkeyAuthentication yes

After this is done, you need to restart the ssh daemon. This is done in OSX with the following commands::
     
     sudo launchctl stop com.openssh.sshd
     sudo launchctl start com.openssh.sshd


Related
--------

*sshconf-*     
        generates the *.ssh/config* file based on env managed node specifications. See *local-vi*
          


EOU

}

ssh--osx-keychain-sock-(){
   find /private/tmp -type s -name 'Listeners' 2>/dev/null 
}
ssh--osx-keychain-sock(){
   $FUNCNAME- 2>/dev/null | head -1
}
ssh--osx-keychain-sock-ids(){
    SSH_AUTH_SOCK=$(ssh--osx-keychain-sock) ssh-add -l
}
ssh--osx-keychain-sock-setup(){
    local SOCK=$(ssh--osx-keychain-sock) 
    SSH_AUTH_SOCK=$SOCK ssh-add -l 2>/dev/null 1>/dev/null && ssh--osx-keychain-sock-persist $SOCK || echo $msg CANNOT FIND SSH_AUTH_SOCK OSX login keychain ssh-agent not running ? 
}
ssh--osx-keychain-sock-export(){
    [ ! -f ~/.ssh-agent-info ] && ssh--osx-keychain-sock-setup
    [   -f ~/.ssh-agent-info ] && source ~/.ssh-agent-info 
}
ssh--osx-keychain-sock-persist(){ 
    $FUNCNAME- $* > ~/.ssh-agent-info
}
ssh--osx-keychain-sock-persist-(){  cat << EOP
# $FUNCNAME 
SSH_AUTH_SOCK=$1 ; export SSH_AUTH_SOCK     
EOP
}


ssh--tunnel(){
  local tag=${1:-N} 
  local port=${2:-8080}

  local sudo
  case $port in
    80) sudo="sudo" ;;
     *) sudo="" ;
  esac
  local cmd="$sudo ssh -fND localhost:$port $tag "

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
    echo -n  # nop
}

ssh--info-checkpid(){
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
  # CAUTION THIS CAN EASILY BREAK BACKUPS
  echo $SSH_INFOFILE
  #echo $HOME/.ssh-agent-info
}

ssh--agent-ok-(){
  ## agent must be running and hold some identities to be "ok"
   ssh-add -l >& /dev/null
}
ssh--agent-ok(){ $FUNCNAME- && echo y || echo n ; }

ssh--agent-notify(){
  python $(env-home)/python/pipemail.py $(local-email)
}

ssh--agent-monitor(){
   local msg="=== $FUNCNAME :"
   local user=${1:-$USER}
   local tmp=/tmp/$USER/env/$FUNCNAME.out
   mkdir -p $(dirname $tmp)  
   ssh--agent-check-user $user > $tmp 
   local rc=$?
   cat $tmp
   if [ "$rc" == "1" ]; then 
       echo $msg rc $rc sending notification email  	   
       cat $tmp | ssh--agent-notify 	    
   else
       echo $msg rc $rc	    
   fi 	    
}

ssh--agent-check-user(){
   local user=${1:-$USER}
   local msg="=== $FUNCNAME :"
   local pid=$(pgrep -u $user ssh-agent)
   echo $msg $(date)
   if [ "$pid" == "" ]; then 
       echo $msg ssh-agent for user $user NOT FOUND
       return 1
   else
       echo $msg ssh-agent for user $user has pid $pid	    
   fi 	   
   return 0
}

ssh--agent-check(){
  [ -z "$SSH_AGENT_PID" ]  && return 1
  [ -z "$SSH_AUTH_SOCK" ]  && return 2 
  ## checks the process is running ... does not kill 
  kill -0 $SSH_AGENT_PID 2>/dev/null 
}

ssh--envdump(){
  local msg="=== $FUNCNAME :"
  cat << EOD
$msg

    USER     $USER
    HOME     $HOME
    NODE     $NODE 
    NODE_TAG $NODE_TAG

    SSH_INFOFILE  : $SSH_INFOFILE
    SSH_AGENT_PID : $SSH_AGENT_PID
    SSH_AUTH_SOCK : $SSH_AUTH_SOCK
EOD

}

ssh--agent-ps(){ ps aux | grep ssh-agent ; }
ssh--agent-stop(){ pkill ssh-agent ; rm  $(ssh--infofile) ; }


ssh--agent-start-notes(){ cat << EON

Attempts to do this in more involved manner
run into security related errors::

    ssh_askpass: exec(/usr/libexec/ssh-askpass): No such file or directory

As OSX needs to prompt for a password but that 
seems not to work in involved scripts.

* https://github.com/markcarver/mac-ssh-askpass

EON
}


ssh--agent-start-(){
    local info=$(ssh--infofile)
    ssh-agent > $info && perl -pi -e 's/echo/#echo/' $info && chmod 0600 $info 
    echo ===== sourcing the info for the agent $info
    . $info
}
ssh--agent-start(){
    ssh--agent-start-   
    echo $FUNCNAME : adding identities to the agent 
    ssh-add $HOME/.ssh/id_dsa $HOME/.ssh/id_rsa
    ssh-add -l
}
ssh--agent-start-alt(){
    ssh--agent-start-   
    echo $FUNCNAME : adding identities to the agent 
    ssh-add -D  # delete all identities as somehow the defaults keep getting added
    ssh-add $HOME/.ssh/alt_dsa
    ssh-add -l
}
ssh--agent-start-dsa(){
    ssh--agent-start-   
    echo $FUNCNAME : adding identities to the agent 
    ssh-add $HOME/.ssh/id_dsa
    ssh-add -l
}






ssh--setup(){

  if [ "$(ssh--agent-ok)" == "y" ]; then
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
    cat ~/.ssh/id_{d,r}sa.pub | ssh $X "cat - >> ~/.ssh/authorized_keys"
    ssh $X "chmod 700 .ssh ; chmod 700 .ssh/authorized_keys" 

}

ssh--pk2(){  cat $2 | ssh $1 "cat - >> ~/.ssh/authorized_keys" ; }


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

ssh--neighbour-addkey(){
   local msg="=== $FUNCNAME :"
   local tag=$1
   local key=$2

   [ ! -f "$key" ] && echo $msg ABORT key $key does not exist && return 1
   [ "$(local-tag2node $NODE_TAG)" != "$(local-tag2node $tag)" ] && echo $msg ABORT this can only be done to $tag from node-neighbour sudoer && return 1   

   local user=$(local-tag2user $tag)
   etc-
   local akpath=$(etc-home $user)/.ssh/$(ssh--key2ak $key)
   sudo mkdir -p $(dirname $akpath)
   local haskey=$(sudo grep "$(cat $key)" $akpath > /dev/null && echo YES || echo NO)
   case $haskey in
     YES) echo $msg $key is already placed in   $akpath && return 0 ;;
      NO) echo $msg proceeding to append key to $akpath && sudo bash -c "cat $key >> $akpath "   ;;
   esac

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

ssh--short-key(){      echo $HOME/.ssh/$1.id_rsa.pub ; }
ssh--short-len(){      echo 6 ; }


ssh--grab-key(){
   local msg="=== $FUNCNAME :"
   
   local path=${1:-$(ssh--designated-key)}
   local tag=$(ssh--key2tag $path)
   local base=$(ssh--key2base $path)

   #echo $msg path $path tag $tag base $base 
   [ -f $path    ] && echo $msg $path is already present && return 0
   [ ! -f $path ]  && scp $tag:~/.ssh/$base $path  
   [ ! -f $path ]  && echo $msg FAILED to grab $path from $tag ... you need to ssh--keygen on $tag  && return 1
}


ssh--key2tag(){
   local msg="=== $FUNCNAME :"
   local path=${1:-$(ssh--designated-key)}
   if [ ${#path} -lt $(ssh--short-len) ] ; then
      echo $path
   else 
      local name=$(basename $path)
      local dtag=${name/.*/}
      echo $dtag
   fi 
}

ssh--key2base(){
   local msg="=== $FUNCNAME :"
   local path=${1:-$(ssh--designated-key)}
   if [ ${#path} -lt $(ssh--short-len) ] ; then
      echo $(ssh--key2base $(ssh--short-key $path))
   else 
      local name=$(basename $path)
      local dtag=$(ssh--key2tag $path)
      local n=$(( ${#dtag} + 1 ))
      local base=${name:$n}
      echo $base
  fi 
}

ssh--key2path(){
   local msg="=== $FUNCNAME :"
   local path=${1:-$(ssh--designated-key)}
   if [ ${#path} -lt $(ssh--short-len) ] ; then
      echo $(ssh--short-key $path)
   else
      echo $path
   fi 
}


ssh--ishub-(){ [ "$NODE_TAG" == "$(ssh--hubtag)" ] && return 0 || return 1 ; }
ssh--hubtag(){  echo G ; }

ssh--distribute-key(){

   local msg="=== $FUNCNAME :"
   ! ssh--ishub- && echo $msg ABORT this must be run from hub node $(ssh--hubtag) && return 1 

   local path=${1:-$(ssh--designated-key)}
   if [ ${#path} -lt $(ssh--short-len) ]; then
      path=$(ssh--key2path $path)
   fi

   shift
   local tags=$*

   local ans
   read -p "$msg $path to nodes : $tags , enter YES to proceed " ans
   [ "$ans" != "YES" ]    && echo $msg skipping && return 1
  
   if [ ! -f "$path" ]; then
      echo $msg key $path does not exist ... attempting to grab it 
      ssh--grab-key $path    
      [ ! -f "$path" ] && echo $msg ABORT : FAILED TO GRAB KEY ... && return 1
   fi

   local tag
   for tag in $tags ; do
      #echo $msg ... $tag $path
      case ${SSH__DISTRIBUTE_KEY:-add} in 
          INI) ssh--inikey  $tag $path ;;
          add) ssh--addkey  $tag $path ;;
      esac
   done
}

ssh--retract-key(){

   local msg="=== $FUNCNAME :"
   ! ssh--ishub- && echo $msg ABORT this must be run from hub node $(ssh--hubtag) && return 1 

   local path=$1
   shift
   local tags=$*

   local ans
   read -p "$msg $path to nodes : $tags , enter YES to proceed " ans
   [ "$ans" != "YES" ]    && echo $msg skipping && return 1

   local tag
   for tag in $tags ; do
      #echo $msg ... $tag $path
      ssh--delkey  $tag $path 
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

ssh--designated-tags(){
   local serverkey=${1:-$(ssh--designated-key)}
   local dtag=$(ssh--key2tag $serverkey)
   local tags=$(local-backup-tag $dtag)
   echo $tags
}


ssh--server-authkeys(){

   local msg="=== $FUNCNAME :"
   ! ssh--ishub- && echo $msg ABORT this must be run from hub node $(ssh--hubtag) && return 1 


   local serverkey=$(ssh--designated-key)
   local dtag=$(ssh--key2tag $serverkey)
   local tags=$(ssh--designated-tags $serverkey)

   local ans
   read -p "$msg grab + distribute server public key $serverkey to server backup nodes $tags ... YES to proceed " ans
   [ "$ans" != "YES" ] && echo $msg skipping && return 0
   echo $msg proceeding 

   ssh--grab-key $serverkey    
   [ ! -f "$serverkey" ] && echo $msg ABORT : FAILED TO GRAB KEY ... && return 1

   ssh--distribute-key $serverkey $tags 

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

ssh--pwauth(){

  local pwauth=${1:-no}

  local msg="=== $FUNCNAME :"
  local cfg=/etc/ssh/sshd_config
  local tmp=/tmp/$FUNCNAME/env && mkdir -p $tmp  
  local tfg=$tmp/$(basename $cfg)

  echo $msg ... password needed for access to $cfg ...
  type $FUNCNAME

  sudo cp $cfg $tfg
  sudo perl -pi -e "s,^(PasswordAuthentication) (\S*),\$1 $pwauth," $tfg   
  sudo diff $cfg $tfg 

  local ans
  read -p "$msg proceed with this config change to $cfg ? YES to continue " ans
  [ "$ans" != "YES" ] && echo $msg skipping && sudo rm $tfg && return 0
 
  sudo cp $tfg $cfg
  sudo rm $tfg

  sudo /sbin/service sshd reload


}
