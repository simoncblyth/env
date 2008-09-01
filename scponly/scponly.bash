
scponly-usage(){

  cat << EOU

       http://sublimation.org/scponly/wiki/index.php/Main_Page
       https://lists.ccs.neu.edu/pipermail/scponly/
       
       
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
       

   
         scponly-user    : $(scponly-user)    
         scponly-chown <username>   
             set ownership of the users home directory, default to the user during setup,
             lockdown with
                   scponly-chown root
             
                                                                
         scponly-chsh  <shell-path>
              set the shell, default to $(scponly-bin)
              open up with 
                    scponly-chsh /bin/bash
         
         
                                                 
         scponly-permissions
              root ownership for lockdown to prevent subverting scponly via copy in
              of .ssh command files but then must open up a bit to allow ssh to read the keys

                                                                                                                                                                                                                          
                                                                               
               
EOU


}


scponly-env(){
   elocal-  
}

scponly-name(){ echo scponly-4.8 ;}
scponly-tgz(){  echo $(scponly-name).tgz ; }
scponly-url(){  echo http://nchc.dl.sourceforge.net/sourceforge/scponly/$(scponly-tgz) ; }
scponly-dir(){  echo $LOCAL_BASE/env/scponly ; }
scponly-builddir(){ echo $(scponly-dir)/build/$(scponly-name) ; }
scponly-cd(){ cd $(scponly-builddir); }
scponly-bin(){  echo $(scponly-dir)/bin/scponly ; }


scponly-user(){  echo dayabayscp ; }



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



scponly-ls(){

  local user=$(scponly-user) 

  sudo ls -la  /home/$user
  sudo ls -la  /home/$user/.ssh

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


