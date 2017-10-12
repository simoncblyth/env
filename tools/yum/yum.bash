# === func-gen- : tools/yum/yum fgp tools/yum/yum.bash fgn yum fgh tools/yum
yum-src(){      echo tools/yum/yum.bash ; }
yum-source(){   echo ${BASH_SOURCE:-$(env-home)/$(yum-src)} ; }
yum-vi(){       vi $(yum-source) ; }
yum-env(){      elocal- ; }
yum-usage(){ cat << EOU


::

    [root@localhost yum.repos.d]# ldd /usr/bin/python
    [root@localhost yum.repos.d]# yum whatprovides /lib64/libc.so.6
    Loaded plugins: priorities, refresh-packagekit, security
    ...
    glibc-2.12-1.149.el6.x86_64 : The GNU libc libraries
    Repo        : sl-security
    Matched from:
    Filename    : /lib64/libc.so.6



EOU
}
yum-dir(){ echo $(local-base)/env/tools/yum/tools/yum-yum ; }
yum-cd(){  cd $(yum-dir); }
yum-mate(){ mate $(yum-dir) ; }
yum-get(){
   local dir=$(dirname $(yum-dir)) &&  mkdir -p $dir && cd $dir

}
