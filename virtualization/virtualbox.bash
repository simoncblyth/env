# === func-gen- : virtualbox/virtualbox fgp virtualbox/virtualbox.bash fgn virtualbox fgh virtualbox
virtualbox-src(){      echo virtualization/virtualbox.bash ; }
virtualbox-source(){   echo ${BASH_SOURCE:-$(env-home)/$(virtualbox-src)} ; }
virtualbox-vi(){       vi $(virtualbox-source) ; }
virtualbox-env(){      elocal- ; }
virtualbox-usage(){ cat << EOU

VIRTUALBOX
===========

* https://www.virtualbox.org/
* https://www.virtualbox.org/manual/ch01.html
* http://www.zdnet.com/virtualisation-suites-compared_p5-7000001456/

* http://download.virtualbox.org/virtualbox/rpm/rhel/
* http://www.dizwell.com/2012/01/16/virtualbox-installations-on-scientific-linux/
* http://wiki.centos.org/HowTos/Virtualization/VirtualBox


Trac Installation testing on SL 5.9
--------------------------------------

belle7 is SL 5.1, intend to migrate Trac+SVN to SL 5.9
Virtualbox could provide a convenient way of debugging 
the installation in a virtual machine before doing it for real.

* https://www.virtualbox.org/wiki/Guest_OSes

   * RHEL5, CentOS 5 (32/64-bit), Works with Additions, 5.3+ Recommended if using VirtIO.


CUDA from virtual OS ?
------------------------

* :google:`virtualbox cuda passthrough`
* https://forums.virtualbox.org/search.php?keywords=cuda
* https://www.virtualbox.org/manual/ch09.html#pcipassthrough

   * using GPU from guest OS is not currently possible (?)

* http://stackoverflow.com/questions/5298453/cuda-opencl-within-a-virtual-machine-hypervisor



EOU
}
virtualbox-dir(){ echo $(local-base)/env/virtualbox/virtualbox-virtualbox ; }
virtualbox-cd(){  cd $(virtualbox-dir); }
virtualbox-mate(){ mate $(virtualbox-dir) ; }
virtualbox-get(){
   local dir=$(dirname $(virtualbox-dir)) &&  mkdir -p $dir && cd $dir

}
