# === func-gen- : virtualization/xen fgp virtualization/xen.bash fgn xen fgh virtualization
xen-src(){      echo virtualization/xen.bash ; }
xen-source(){   echo ${BASH_SOURCE:-$(env-home)/$(xen-src)} ; }
xen-vi(){       vi $(xen-source) ; }
xen-env(){      elocal- ; }
xen-usage(){ cat << EOU

XEN HYPERVISOR
================

* http://en.wikipedia.org/wiki/Xen
* :google:`xen virtualbox comparison`
* http://www.zdnet.com/virtualisation-suites-compared-7000001456/

Xen is a hypervisor providing services that allow multiple computer operating
systems to execute on the same computer hardware concurrently.



EOU
}
xen-dir(){ echo $(local-base)/env/virtualization/virtualization-xen ; }
xen-cd(){  cd $(xen-dir); }
xen-mate(){ mate $(xen-dir) ; }
xen-get(){
   local dir=$(dirname $(xen-dir)) &&  mkdir -p $dir && cd $dir

}
