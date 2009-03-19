scponly-src(){     echo scponly/scponly.bash ; }
scponly-source(){  echo $(env-home)/$(scponly-src) ; }
scponly-srcurl(){     echo $(env-url)/$(scponly-src) ; }
scponly-vi(){      vi $(scponly-source) ; }
scponly-usage(){

  cat << EOU


    REFERENCE
      
       http://sublimation.org/scponly/wiki/index.php/Main_Page
       https://lists.ccs.neu.edu/pipermail/scponly/
        
     
    CONFIG 
   
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
              scponly-ls
              grep /etc/passwd $(scponly-user)




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
   sudo ls -la  $(scponly-home)
   sudo ls -la  $(scponly-home)/.ssh
   grep /etc/passwd $(scponly-user)
}


scponly-adduser(){
   local user=$(scponly-user) 
   which useradd
   echo sudo useradd -d $(dirname $HOME)/$user -s $(scponly-bin) $user 
}

scponly-chown(){

   local user=$(scponly-user) 
   local sser=${1:-$user}
   local cmd="sudo chown -R $sser:$sser $(dirname $HOME)/$user"
   echo $cmd
   eval $cmd

}


scponly-chsh(){
    local shell=${1:-$(scponly-bin)}
    local cmd="sudo chsh -s $shell $(scponly-user)"
    echo $cmd
    eval $cmd
}


scponly-permissions(){

   local user=$(scponly-user)
   
   ## must lock down to prevent subventing scponly via copyin of .ssh command files
   sudo chown -R root:root  /home/$user

   ## but must open up a bit to allow ssh to read the keys
   
   sudo chmod 755 /home/$user
   sudo chmod 755 /home/$user/.ssh


}


scponly-log(){
   sudo tail -f /var/log/messages
}


scponly-target(){

   local user=$(scponly-user)
   local dir=$SCM_FOLD/backup/dayabay
   
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




