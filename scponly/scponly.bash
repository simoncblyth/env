scponly-src(){     echo scponly/scponly.bash ; }
scponly-source(){  echo $(env-home)/$(scponly-src) ; }
scponly-srcurl(){     echo $(env-url)/$(scponly-src) ; }
scponly-vi(){      vi $(scponly-source) ; }
scponly-usage(){

  cat << EOU


    REFERENCE
      
       http://sublimation.org/scponly/wiki/index.php/Main_Page
       https://lists.ccs.neu.edu/pipermail/scponly/
        


    PROCEDURE  

       0) node tagging 

           Create a tag for the new locked down node, eg S S2
           in sshconf-vi/local-vi and generate the .ssh/config on the needed scp 
           source nodes

        
       1) create a new user initially without restrictions

            On the fresh node...  in an unrestricted sudoer account
               scponly-
               scponly-useradd     
                     create new user and set the password, initially with /bin/bash

               scponly-passkeys    
                     create .ssh for the user and pass them the keys 
                 
       2) verify passwordless login and scp from remote node (the pubkey of which was passed)

               ssh SC2
               scponly-test SC2

       3)  lockdown : switch to restricted shell and permissions

               scponly-lockdown
                      switch the shell and permissions 

       4)  test that logins fails and scps still work passwordlessly 

               ssh SC2  ... should fail 
               scponly-test SC2

       5)  setup backup target dirs 
               
               on the target node from unrestricted sudoer account ... 
                   scm-backup-
                   dir=`scm-backup-dir`/dayabay # for backup tarballs originating from dayabay 
                   sudo mkdir -p $dir      
                   sudo chown dayabayscp.dayabayscp $dir


     
    CONFIG 


         scponly-create

                 
         scponly-user    : $(scponly-user)    
         scponly-chown <username>   
             set ownership of the users home directory, default to the user during setup,
             lockdown with
                   scponly-chown root
             
                                                                
         scponly-chsh  <shell-path>
                    scponly-chsh               # default sets shell to $(scponly-bin)
                    scponly-chsh /bin/bash
        
         scponly-log 
                tail the system log 
         
                                                 
         scponly-permissions
              root ownership for lockdown to prevent subverting scponly via copy in
              of .ssh command files but then must open up a bit to allow ssh to read the keys

         scponly-info


         scponly-lstags
              loop over ssh--tags : $(ssh--tags) greping /etc/shells for scponly 


                  




       NOTES 

           Cannot "ssh--putkey" to a locked down account ... so transfer from an account on 
           the same node like :
                sudo bash -c "cat .ssh/authorized_keys2 >> ../dayabayscp/.ssh/authorized_keys2 "
                sudo cat  ../dayabayscp/.ssh/authorized_keys2  



       BUILD RELATED 

         scponly-name     : $(scponly-name)  
         scponly-url      : $(scponly-url)  
         scponly-dir      : $(scponly-dir)  
         scponly-builddir : $(scponly-builddir)
        
         scponly-cd       : to builddir
 
         scponly-get/configure/install :
         scponly-configure :
               using options
                  scponly-opts :  $(scponly-opts)
              
              
              configure: WARNING: read the SECURITY document before enabling rsync compatibility
 


        scponly-test 
               the restricted user generally does not have permission to write into 
               their home directory

               but they must have permission to read the public keys in $HOME/.ssh/authorized_keys2              



       to locked down account on cms01 ...

simon:scponly-test blyth$ scponly-test S
=== scponly-test : scp scponly-test.txt S:/tmp/scponly-test.txt
Scientific Linux CERN SLC release 4.7 (Beryllium)
scponly-test.txt                                                                                                                     100%   35     0.0KB/s   00:00    


       to locked down account on grid1 ...
    
simon:scponly-test blyth$ scponly-test S2
=== scponly-test : scp scponly-test.txt S2:/tmp/scponly-test.txt
Scientific Linux CERN Release 3.0.8 (SL)
/disk/d3/dayabay/local/env/scponly/bin/scponly: Permission denied
lost connection

      the locked down user on grid1 has a ginormous uid ... due to horrible number of users on grid1 :
dayabayscp:x:45052:45052::/home/hep/dayabayscp:/disk/d3/dayabay/local/env/scponly/bin/scponly
[blyth@grid1 blyth]$ 
[blyth@grid1 blyth]$ cat /etc/passwd | wc
   1053    5941   77391




EOU


}


scponly-env(){   elocal-  ; }
scponly-user(){  echo dayabayscp ; }
scponly-home(){
   local t=${1:-$NODE_TAG}
   case $t in
            G1) echo /home/hep/$(scponly-user $t) ;;
           C|S) echo /home/$(scponly-user $t) ;;
         C2|S2) echo /home/$(scponly-user $t) ;;
          N|SN) echo /home/$(scponly-user $t) ;;
             *) echo /home/$(scponly-user $t) ;;
   esac
}


scponly-info(){

   cat << EOI

     scponly-bin   : $(scponly-bin)
     scponly-home  : $(scponly-home)


EOI

}



scponly-put-ihepkey(){
  local tag=$1
  [ "$NODE_TAG" != "C" ] && echo $msg the key resides on node C .. not $NODE_TAG && return 1
  head -1 $(scponly-home)/.ssh/authorized_keys2 | ssh $tag "cat - >> ~/.ssh/authorized_keys2";
}



scponly-check(){

   local msg="=== $FUNCNAME :"

   echo $msg scponly-bin $(scponly-bin)
   sudo ls -la  $(scponly-bin)
   echo $msg scponly-home : $(scponly-home)
   sudo ls -la  $(scponly-home)
   echo $msg scponly-home/.ssh : $(scponly-home)/.ssh
   sudo ls -la  $(scponly-home)/.ssh

   echo $msg check for user $(scponly-user) in /etc/passwd
   grep $(scponly-user) /etc/passwd

   echo $msg restricted scponly shell $(scponly-bin) should appear in the list
   tail -5 /etc/shells

   echo $msg target dirs for backup tarballs should exist and be owned by $(scponly-user)
   local target
   for target in $(scponly-target-list) ; do  
      echo $msg looking at target $target 
      sudo ls -la  $target
   done 


   scponly-keys

}



scponly-testexit(){
   ls -al $(scponly-bin)
   local msg="=== $FUNCNAME : "
   sudo -u $(scponly-user)  $(scponly-bin) 
   [ "$?" == "1" ] && echo $msg expected behaviour || echo $msg UNEXPECTED rc 
}


scponly-test(){
   local tag=$1 
   local msg="=== $FUNCNAME :"
   local tmp=/tmp/env/$FUNCNAME && mkdir -p $tmp
   local name=$FUNCNAME-from-${NODE_TAG}-to-$tag.txt
   echo $name > $tmp/$name
   local cmd="scp $tmp/$name $tag:/tmp/$name"
   echo $msg $cmd
   eval $cmd 
}


scponly-test-all(){

  local msg="=== $FUNCNAME :"
  local tag
  for tag in $(local-scponly-tags) ; do

     echo $msg for tag $tag 
     local-taginfo $tag

     case $tag in
       SC2) echo $msg skipping $tag ;;
         *) scponly-test $tag       ;;
     esac
  done
}


scponly-lockdown(){
    scponly-addshell
    scponly-chsh $(scponly-bin)
    scponly-permissions root
}

scponly-openup(){
    scponly-chsh /bin/bash
    scponly-permissions $(scponly-user)
}



scponly-passkeys(){
    local msg="=== $FUNCNAME :"
    [ "$HOME" == "$(scponly-home)" ] && echo $msg ABORT ... this is not to be done by the restricted user && return 1
    ssh--createdir $(scponly-home)
    sudo bash -c "cat $HOME/.ssh/authorized_keys2 >> $(scponly-home)/.ssh/authorized_keys2 " 
}

scponly-keys(){
    sudo cat $(scponly-home)/.ssh/authorized_keys2
}

scponly-useradd(){
   local msg="=== $FUNCNAME :"
   [ -d "$(scponly-home)" ] && echo $msg ABORT users home $(scponly-home) exists already && return 1
   local cmd="sudo /usr/sbin/useradd -d $(scponly-home) -s /bin/bash $(scponly-user)  "
   echo $msg $cmd
   eval $cmd

   local cme="sudo passwd $(scponly-user)"
   echo $msg $cme   NB ... the password for the new account ... not for sudo access 
   eval $cme

}

scponly-userdel(){
   local msg="=== $FUNCNAME :"
   local cmd="sudo /usr/sbin/userdel -r $(scponly-user) "
   echo $msg $cmd
   eval $cmd
}


scponly-chsh(){
    local shell=${1:-$(scponly-bin)}
    local cmd="sudo chsh -s $shell $(scponly-user)"
    echo $cmd
    eval $cmd
}






scponly-permissions(){

   local user=${1:-root} 

   ## must lock down to prevent subventing scponly via copyin of .ssh command files
   sudo chown -R $user:$user $(scponly-home)

   ## but must open up a bit to allow ssh to read the keys

   ## chmod u+rwx,go+rx 
   
   sudo chmod 755 $(scponly-home)
   sudo chmod 755 $(scponly-home)/.ssh
   sudo chmod go+r $(scponly-home)/.ssh/authorized_keys2

   scponly-check

}


scponly-log(){
   sudo tail -f /var/log/messages
}



scponly-target-list(){
  cat << EOT
$(local-scm-fold)/backup/dayabay 
EOT
}

scponly-setup-targets(){
   local target
   for target in $(scponly-target-list) ; do  
      scponly-target $target
   done 
}




scponly-target(){
   local msg="=== $FUNCNAME :" 
   local user=$(scponly-user)
   local dir=$1
   echo $msg setup target dir $dir  to catch scp backup tarballs 
   sudo mkdir -p $dir
   sudo chown -R $user:$user $dir
}





## BUILD RELATED 


scponly-name(){ echo scponly-4.8 ;}
scponly-tgz(){  echo $(scponly-name).tgz ; }
scponly-url(){  echo http://nchc.dl.sourceforge.net/sourceforge/scponly/$(scponly-tgz) ; }
scponly-dir(){  echo $LOCAL_BASE/env/scponly ; }
scponly-builddir(){ echo $(scponly-dir)/build/$(scponly-name) ; }
scponly-cd(){ cd $(scponly-builddir); }
scponly-bin(){  echo $(scponly-dir)/bin/scponly ; }


scponly-get(){

    local dir=$(scponly-dir)
    local nam=$(basename $dir)
    cd $(dirname $dir)
    $SUDO mkdir -p $nam
    $SUDO chown $USER $nam
    cd $nam
    
   local tgz=$(scponly-tgz)

    [ ! -f $tgz ] && curl -O $(scponly-url)
    mkdir -p  build
    [ ! -d build/$(scponly-name) ] && tar -C build -zxvf $tgz  
}



scponly-opts(){
   echo  --enable-chrooted-binary  --disable-winscp-compat --disable-sftp   --enable-scp-compat   --enable-rsync-compat  
}

scponly-configure(){

   scponly-cd

   ./configure -h
   ./configure --prefix=$(scponly-dir) $(scponly-opts)

}

scponly-install(){
   
   scponly-cd
   make 
   sudo make install

}

scponly-wipe(){
   local msg="=== $FUNCNAME :"
   cd $(scponly-dir)  
   [ ! -f $(scponly-tgz) ] && echo $msg no tgz smth wrong && return 1
   
   rm -rf build man etc bin 
   
 
}

scponly-addshell(){
   local msg="=== $FUNCNAME :"
   local sh=$(scponly-bin)
   grep $sh /etc/shells && echo $msg shell $sh is already present && return 0

   local cmd=$(cat << EOC
sudo bash -c "echo $sh >> /etc/shells" 
EOC)

   echo $msg $cmd
   eval $cmd
   tail  /etc/shells
}

scponly-has-shell(){ ssh--cmd $1 grep scponly /etc/shells ; }
scponly-has-user(){  ssh--cmd $1 grep scponly /etc/passwd ; }

scponly-lstags(){
   local tag
   for tag in $(ssh--tags) ; do
       printf " %4s %s %s \n" $tag $(scponly-has-shell $tag) $(scponly-has-user $tag)  
   done
}







