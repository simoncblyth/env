
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
       

          copy with setting of owner and group to that for root 
              /usr/bin/install -c  -o 0 -g 0 scponly $LOCAL_BASE/env/scponly/bin/scponly
       
        ${INSTALL} -o 0 -g 0 scponly ${DESTDIR}${bindir}/scponly
                   
                         
         
                                     
      Manual steps ..
         1)  introduce the scponly shell to the system 
         
                 cat /etc/shells
                 sudo bash -c "echo $(scponly-bin) >> /etc/shells "
         
         2)  add user http://www.linfo.org/useradd.html

                  cat /etc/passwd 
                  sudo  /usr/sbin/useradd -d $(dirname $HOME)/dayabayscp -s $(scponly-bin) dayabayscp                                                                                    
                 
         3) set the passwrd
         
                 sudo passwd dayabayscp
Password:
Changing password for user dayabayscp.
New UNIX password: 
Retype new UNIX password: 
passwd: all authentication tokens updated successfully. 
                
        4) change ownership of /home/dayabayscp
                 
               sudo chown root:root /home/dayabayscp             
               
            from scponly README/INSTALL :
                It is very important that the user's home directory 
                be unwritable by the user, as a writable homedir 
                will make it possible for users to subvert scponly 
                by modifying ssh configuration files.                                                                                                                                     
               
        
               
        test scp to /tmp
        
    simon:~ blyth$ scp yum.list dayabayscp@cms01.phys.ntu.edu.tw:/tmp/
The authenticity of host 'cms01.phys.ntu.edu.tw (140.112.101.190)' can't be established.
RSA key fingerprint is c2:3f:91:87:91:21:98:1b:91:fe:c2:c4:1f:5b:de:6c.
Are you sure you want to continue connecting (yes/no)? yes
Warning: Permanently added 'cms01.phys.ntu.edu.tw' (RSA) to the list of known hosts.
Scientific Linux CERN SLC release 4.6 (Beryllium)
dayabayscp@cms01.phys.ntu.edu.tw's password: 
Could not chdir to home directory /home/dayabayscp: Permission denied
yum.list            
        
                                    
          test login attempt                           
                                    
               
        simon:~ blyth$ ssh S
Scientific Linux CERN SLC release 4.6 (Beryllium)
dayabayscp@140.112.101.190's password: 
Could not chdir to home directory /home/dayabayscp: Permission denied
Connection to 140.112.101.190 closed.  
                    
               
               
         next ... key setup     
               
               
                   
               
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

   
   grep -v $bin /etc/shells 


}

