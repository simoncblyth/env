# === func-gen- : tools/dnf/dnf fgp tools/dnf/dnf.bash fgn dnf fgh tools/dnf src base/func.bash
dnf-source(){   echo ${BASH_SOURCE} ; }
dnf-edir(){ echo $(dirname $(dnf-source)) ; }
dnf-ecd(){  cd $(dnf-edir); }
dnf-dir(){  echo $LOCAL_BASE/env/tools/dnf/dnf ; }
dnf-cd(){   cd $(dnf-dir); }
dnf-vi(){   vi $(dnf-source) ; }
dnf-env(){  elocal- ; }
dnf-usage(){ cat << EOU

DNF : next gen YUM
=====================

DNF has been the default package manager for Fedora since version 22 which was
released in May 2015.[6] The libdnf library is used as a package backend in
PackageKit.[9] DNF is also available as an alternate package manager for Mageia
Linux since version 6. It may become the default sometime in the future.[10]


Based on libsolv

* https://github.com/openSUSE/libsolv
* https://en.opensuse.org/openSUSE:Libzypp_satsolver




* https://www.ostechnix.com/dnf-command-examples-beginners/

As of Fedora 22, yum has been replaced with DNF, so you don’t need to install
it if you’re on Fedora. On CentOS 7 and RHEL 7, you can install it as described
in the tutorial given below.


* https://access.redhat.com/discussions/3387801

Michael Young

It is new in RHEL 7.6 - see
https://access.redhat.com/documentation/en-us/red_hat_enterprise_linux/7/html/7.6_release_notes/technology_previews_system_and_subscription_management
. It doesn't say so in the article but /usr/bin/yum4 is just a symbolic link to
dnf-2 , as is /usr/bin/dnf .





EOU
}
dnf-get(){
   local dir=$(dirname $(dnf-dir)) &&  mkdir -p $dir && cd $dir

}
