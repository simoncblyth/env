# === func-gen- : git/libgit2 fgp git/libgit2.bash fgn libgit2 fgh git
libgit2-src(){      echo git/libgit2.bash ; }
libgit2-source(){   echo ${BASH_SOURCE:-$(env-home)/$(libgit2-src)} ; }
libgit2-vi(){       vi $(libgit2-source) ; }
libgit2-env(){      elocal- ; }
libgit2-usage(){ cat << EOU

LibGit2
=========

https://libgit2.github.com

libgit2 is a portable, pure C implementation of the Git core methods provided
as a re-entrant linkable library with a solid API, allowing you to write native
speed custom Git applications in any language which supports C bindings

* this means git can go anywhere
* maybe should migrate my repos  from hg to git ?


Macports on fresh system
--------------------------

::

    epsilon:home blyth$ sudo port install -v libgit2
    --->  Computing dependencies for libgit2
    The following dependencies will be installed: 
     bzip2
     curl
     curl-ca-bundle
     gettext
     glib2
     libedit
     libffi
     libiconv
     libidn2
     libpsl
     libssh2
     libunistring
     ncurses
     openssl
     pcre
     zlib
    Continue? [Y/n]: Y
    --->  Fetching archive for curl-ca-bundle
    --->  Attempting to fetch curl-ca-bundle-7.59.0_0.darwin_17.noarch.tbz2 from http://kmq.jp.packages.macports.org/curl-ca-bundle
    --->  Attempting to fetch curl-ca-bundle-7.59.0_0.darwin_17.noarch.tbz2.rmd160 from http://kmq.jp.packages.macports.org/curl-ca-bundle
    --->  Installing curl-ca-bundle @7.59.0_0
    --->  Activating curl-ca-bundle 
    ...

::

    epsilon:home blyth$ port contents libgit2
    Port libgit2 contains:
      /opt/local/include/git2.h
      /opt/local/include/git2/annotated_commit.h
      /opt/local/include/git2/attr.h
      /opt/local/include/git2/blame.h
      ...
      /opt/local/include/git2/version.h
      /opt/local/include/git2/worktree.h
      /opt/local/lib/libgit2.0.26.3.dylib
      /opt/local/lib/libgit2.26.dylib
      /opt/local/lib/libgit2.dylib
      /opt/local/lib/pkgconfig/libgit2.pc
    epsilon:home blyth$ 



EOU
}
libgit2-dir(){ echo $(local-base)/env/git/git-libgit2 ; }
libgit2-cd(){  cd $(libgit2-dir); }
libgit2-mate(){ mate $(libgit2-dir) ; }
libgit2-get(){
   local dir=$(dirname $(libgit2-dir)) &&  mkdir -p $dir && cd $dir

   sudo port install -v libgit2  

}
