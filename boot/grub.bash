# === func-gen- : boot/grub fgp boot/grub.bash fgn grub fgh boot src base/func.bash
grub-source(){   echo ${BASH_SOURCE} ; }
grub-edir(){ echo $(dirname $(grub-source)) ; }
grub-ecd(){  cd $(grub-edir); }
grub-dir(){  echo $LOCAL_BASE/env/boot/grub ; }
grub-cd(){   cd $(grub-dir); }
grub-vi(){   vi $(grub-source) ; }
grub-env(){  elocal- ; }
grub-usage(){ cat << EOU

GRUB
======

* https://en.wikipedia.org/wiki/GNU_GRUB

* https://wiki.centos.org/HowTos/Grub2


/boot
-------

::

    [blyth@localhost ~]$ ll /boot/vmlinuz*
    -rwxr-xr-x. 1 root root 5392080 Nov 23  2016 /boot/vmlinuz-3.10.0-514.el7.x86_64
    -rwxr-xr-x. 1 root root 6233824 Jun 27  2018 /boot/vmlinuz-3.10.0-862.6.3.el7.x86_64
    -rwxr-xr-x. 1 root root 5392080 Jul  5  2018 /boot/vmlinuz-0-rescue-f42ac84eae3c4cecb3d493c899463c30
    -rwxr-xr-x. 1 root root 6643904 Mar 18 23:10 /boot/vmlinuz-3.10.0-957.10.1.el7.x86_64


Linux Kernel version numbering 
-------------------------------

* 


Listing what will appear in boot menu
----------------------------------------

grub-ls::

    [blyth@localhost ~]$ sudo awk -F\' '$1=="menuentry " {print i++ " : " $2}' /etc/grub2.cfg
    0 : CentOS Linux (3.10.0-957.10.1.el7.x86_64) 7 (Core)
    1 : CentOS Linux (3.10.0-862.6.3.el7.x86_64) 7 (Core)
    2 : CentOS Linux (3.10.0-514.el7.x86_64) 7 (Core)
    3 : CentOS Linux (0-rescue-f42ac84eae3c4cecb3d493c899463c30) 7 (Core)
    4 : Windows 10 (loader) (on /dev/sda1)

    [blyth@localhost ~]$ uname -a
    Linux localhost.localdomain 3.10.0-862.6.3.el7.x86_64 #1 SMP Tue Jun 26 16:32:21 UTC 2018 x86_64 x86_64 x86_64 GNU/Linux





EOU
}
grub-get(){
   local dir=$(dirname $(grub-dir)) &&  mkdir -p $dir && cd $dir

}

grub-ls(){ sudo awk -F\' '$1=="menuentry " {print i++ " : " $2}' /etc/grub2.cfg ; }
