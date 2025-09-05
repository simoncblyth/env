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






::

    A[blyth@localhost llama_cpp_python]$ dnf search gcc-toolset-??
    ...
    gcc-toolset-12.x86_64 : Package that installs gcc-toolset-12
    gcc-toolset-13.x86_64 : Package that installs gcc-toolset-13
    gcc-toolset-14.x86_64 : Package that installs gcc-toolset-14
    A[blyth@localhost llama_cpp_python]$ 


    A[blyth@localhost llama_cpp_python]$ dnf list installed | grep gcc-toolset-
    gcc-toolset-13-binutils.x86_64                   2.40-21.el9                        @appstream           
    gcc-toolset-13-binutils-gold.x86_64              2.40-21.el9                        @appstream           
    gcc-toolset-13-gcc.x86_64                        13.3.1-2.3.el9                     @appstream           
    gcc-toolset-13-gcc-c++.x86_64                    13.3.1-2.3.el9                     @appstream           
    gcc-toolset-13-libstdc++-devel.x86_64            13.3.1-2.3.el9                     @appstream           
    gcc-toolset-13-runtime.x86_64                    13.0-2.el9                         @appstream           
    gcc-toolset-14-binutils.x86_64                   2.41-4.el9_6.1                     @appstream           
    gcc-toolset-14-gcc.x86_64                        14.2.1-7.1.el9                     @appstream           
    gcc-toolset-14-gcc-c++.x86_64                    14.2.1-7.1.el9                     @appstream           
    gcc-toolset-14-libstdc++-devel.x86_64            14.2.1-7.1.el9                     @appstream           
    gcc-toolset-14-runtime.x86_64                    14.0-1.el9                         @appstream           
    A[blyth@localhost llama_cpp_python]$ 


    A[blyth@localhost llama_cpp_python]$ dnf list installed | grep gcc
    gcc.x86_64                                       11.5.0-5.el9_5.alma.1              @appstream           
    gcc-c++.x86_64                                   11.5.0-5.el9_5.alma.1              @appstream           
    gcc-debuginfo.x86_64                             11.4.1-3.el9.alma.1                @baseos-debuginfo    
    gcc-debugsource.x86_64                           11.4.1-3.el9.alma.1                @baseos-debuginfo    
    gcc-gfortran.x86_64                              11.5.0-5.el9_5.alma.1              @appstream           
    gcc-plugin-annobin.x86_64                        11.5.0-5.el9_5.alma.1              @appstream           
    gcc-toolset-13-binutils.x86_64                   2.40-21.el9                        @appstream           
    gcc-toolset-13-binutils-gold.x86_64              2.40-21.el9                        @appstream           
    gcc-toolset-13-gcc.x86_64                        13.3.1-2.3.el9                     @appstream           
    gcc-toolset-13-gcc-c++.x86_64                    13.3.1-2.3.el9                     @appstream           
    gcc-toolset-13-libstdc++-devel.x86_64            13.3.1-2.3.el9                     @appstream           
    gcc-toolset-13-runtime.x86_64                    13.0-2.el9                         @appstream           
    gcc-toolset-14-binutils.x86_64                   2.41-4.el9_6.1                     @appstream           
    gcc-toolset-14-gcc.x86_64                        14.2.1-7.1.el9                     @appstream           
    gcc-toolset-14-gcc-c++.x86_64                    14.2.1-7.1.el9                     @appstream           
    gcc-toolset-14-libstdc++-devel.x86_64            14.2.1-7.1.el9                     @appstream           
    gcc-toolset-14-runtime.x86_64                    14.0-1.el9                         @appstream           
    libgcc.i686                                      11.5.0-5.el9_5.alma.1              @baseos              
    libgcc.x86_64                                    11.5.0-5.el9_5.alma.1              @baseos              
    libgcc-debuginfo.x86_64                          11.4.1-3.el9.alma.1                @baseos-debuginfo    
    A[blyth@localhost llama_cpp_python]$ 











EOU
}
dnf-get(){
   local dir=$(dirname $(dnf-dir)) &&  mkdir -p $dir && cd $dir

}
