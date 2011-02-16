# === func-gen- : base/redhat fgp base/redhat.bash fgn redhat fgh base
redhat-src(){      echo base/redhat.bash ; }
redhat-source(){   echo ${BASH_SOURCE:-$(env-home)/$(redhat-src)} ; }
redhat-vi(){       vi $(redhat-source) ; }
redhat-env(){      elocal- ; }
redhat-usage(){
  cat << EOU
     redhat-src : $(redhat-src)
     redhat-dir : $(redhat-dir)

     redhat-epel
         Configure yum to have access to EPEL,  Extra Packages for Enterprise Linux  
         http://fedoraproject.org/wiki/EPEL/FAQ#howtouse

     [blyth@belle7 yum.repos.d]$ redhat-vers
     Scientific Linux SL release 5.1 (Boron)

     [blyth@cms01 yum.repos.d]$ redhat-vers
     Scientific Linux CERN SLC release 4.8 (Beryllium)

     Probably best to disable it by default ... 
           /etc/yum.repos.d/epel.d  
     to prevent accidentals  ... see #278 #276

[blyth@belle7 yum.repos.d]$ redhat-epel5
[sudo] password for blyth: 
Retrieving http://download.fedora.redhat.com/pub/epel/5/i386/epel-release-5-3.noarch.rpm
warning: /var/tmp/rpm-xfer.4YsNUc: Header V3 DSA signature: NOKEY, key ID 217521f6
Preparing...                ########################################### [100%]
   1:epel-release           ########################################### [100%]
[blyth@belle7 yum.repos.d]$ 
[blyth@belle7 yum.repos.d]$ 
[blyth@belle7 yum.repos.d]$ rpm -ql epel-release-5-3
/etc/pki/rpm-gpg/RPM-GPG-KEY-EPEL
/etc/yum.repos.d/epel-testing.repo
/etc/yum.repos.d/epel.repo
/usr/share/doc/epel-release-5
/usr/share/doc/epel-release-5/GPL




  Example of searching, with extra EPEL repo included :
  (it is probably wise to not enable EPEL by default)

      yum --enablerepo=epel search pycurl


EOU
}
redhat-dir(){ echo $(local-base)/env/base/base-redhat ; }
redhat-cd(){  cd $(redhat-dir); }
redhat-mate(){ mate $(redhat-dir) ; }
redhat-get(){
   local dir=$(dirname $(redhat-dir)) &&  mkdir -p $dir && cd $dir

}

redhat-vers(){
  cat /etc/redhat-release
}

redhat-epel4(){
   sudo rpm -Uvh http://download.fedora.redhat.com/pub/epel/4/i386/epel-release-4-9.noarch.rpm
}

redhat-epel5(){
  sudo rpm -Uvh http://download.fedora.redhat.com/pub/epel/5/i386/epel-release-5-3.noarch.rpm
}

