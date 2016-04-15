# === func-gen- : virtualbox/vbox fgp virtualbox/vbox.bash fgn vbox fgh virtualbox
vbox-src(){      echo virtualbox/vbox.bash ; }
vbox-source(){   echo ${BASH_SOURCE:-$(env-home)/$(vbox-src)} ; }
vbox-vi(){       vi $(vbox-source) ; }
vbox-env(){      elocal- ; }
vbox-usage(){ cat << EOU

* https://www.virtualbox.org

Presently, VirtualBox runs on Windows, Linux, Macintosh, and Solaris hosts and
supports a large number of guest operating systems including but not limited to
Windows (NT 4.0, 2000, XP, Server 2003, Vista, Windows 7, Windows 8, Windows
10), DOS/Windows 3.x, Linux (2.4, 2.6, 3.x and 4.x), Solaris and OpenSolaris,
OS/2, and OpenBSD.



EOU
}
vbox-dir(){ echo $(local-base)/env/virtualbox/virtualbox-vbox ; }
vbox-cd(){  cd $(vbox-dir); }
vbox-mate(){ mate $(vbox-dir) ; }
vbox-get(){
   local dir=$(dirname $(vbox-dir)) &&  mkdir -p $dir && cd $dir

}
