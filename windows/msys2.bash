# === func-gen- : windows/msys2 fgp windows/msys2.bash fgn msys2 fgh windows
msys2-src(){      echo windows/msys2.bash ; }
msys2-source(){   echo ${BASH_SOURCE:-$(env-home)/$(msys2-src)} ; }
msys2-vi(){       vi $(msys2-source) ; }
msys2-env(){      elocal- ; }
msys2-usage(){ cat << EOU

MSYS2
======

Best introduction to msys2 is the yellowed comment on the below page

* http://stackoverflow.com/questions/25019057/how-are-msys-msys2-and-msysgit-related-to-each-other


Looks promising as a way of bringing a Linux like toolchain
to Windows but yielding native windows binaries, unlike Cygwin
which requires to go thru the Cygwin dll.

* https://msys2.github.io/
* https://sourceforge.net/p/msys2/wiki/MSYS2%20introduction/


* http://www.davidegrayson.com/windev/msys2/

* http://www.mingw.org describes "predecessor" : MSYS


* https://sourceforge.net/p/msys2/wiki/Home/

Introduction
-------------

* https://sourceforge.net/p/msys2/wiki/MSYS2%20introduction/

MSYS2 is an independent rewrite of MSYS, based on modern Cygwin (POSIX
compatibility layer) and MinGW-w64 with the aim of better interoperability with
native Windows software.

MSYS2 consists of three subsystems and their corresponding package repositories, msys2, mingw32, and mingw64.

Every subsystem has an associated "shell", which is essentially a set of
environment variables that allow the subsystems to co-operate properly. These
shells can be invoked using scripts in the MSYS2 installation directory or
shortcuts in the Start menu. The scripts set the MSYSTEM variable and start a
terminal emulator with bash. Bash in turn sources /etc/profile which sets the
environment depending on the value of MSYSTEM.

Use msys2 shell for running pacman, makepkg, makepkg-mingw and for building
POSIX-dependent software that you don't intend to distribute. Use mingw shells
for building native software and other tasks.


Distro
-----------------

MSYS2 packages
~~~~~~~~~~~~~~~

https://github.com/Alexpux/MSYS2-packages

* GPL encumbered + via POSIX emulation, but thats fine for tools


Selection of tools

* bash, bzip2, cmake, curl, expat, git, gzip, libxml2, make, mercurial, openssh, p7zip, python2, python3, subversion, wget



To build these, run msys2_shell.bat then from the bash prompt.::

    cd ${package-name}
    makepkg

To install the built package(s).::

    pacman -U ${package-name}*.pkg.tar.xz

If you don't have the group base-devel installed, please install.::

    pacman -S base-devel


MINGW packages
~~~~~~~~~~~~~~~~~

* https://github.com/Alexpux/MINGW-packages

Selection of "non-tools", component packages all prefixed with "mingw-w64-":

* SDL2, boost, cmake-git, collada-dom-svn, fossil, opencollada-git, python-numpy, python-sphinx, xerces-c, zeromq, assimp, assimp-git



mingw cmake via msys2
-----------------------

* https://news.ycombinator.com/item?id=11200989

RayDonnelly 52 days ago::

    pacman -S mingw-w64-{x86_64,i686}-cmake 

There's you out of the box. You might have tried pacman -S cmake which 
will have gotten you the msys/cmake package which is used for building 
the packages in the msys repo (i.e. those linked to msys-2.0.dll). 
Having to add the mingw-w64-.. prefix is kind of strange, but
you get used to it.


runtime PATH
----------------

Windows treats dll like binaries, for running of .exe to find their
dlls either have to copy the .exe and .dll into the directory with
the dll they depend on or must set PATH. 
It seems windows has no concept of RPATH.



EOU
}
msys2-dir(){ echo $(local-base)/env/windows/windows-msys2 ; }
msys2-cd(){  cd $(msys2-dir); }

msys2-orig-path-(){ cat << EOP
/mingw64/bin
/home/ntuhep/env/bin
/usr/local/bin
/usr/bin
/bin
/c/Windows/System32
/c/Windows
/c/Windows/System32/Wbem
/c/Windows/System32/WindowsPowerShell/v1.0/
/usr/bin/site_perl
/usr/bin/vendor_perl
/usr/bin/core_perl
EOP
}

msys2-test-path-(){ cat << EOP
/mingw64/bin
/usr/local/opticks/bin
/usr/local/opticks/lib
/home/ntuhep/env/bin
/usr/local/bin
/usr/bin
/bin
/c/Windows/System32
/c/Windows
/c/Windows/System32/Wbem
/c/Windows/System32/WindowsPowerShell/v1.0/
EOP
}

msys2-path(){
   local name=${1:-orig} 
   local path="$(echo $(msys2-${name}-path-))"
   echo ${path// /:}
}

msys2-export()
{
   export PATH=$(msys2-path ${1:-test})
}

