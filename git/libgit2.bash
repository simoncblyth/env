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


EOU
}
libgit2-dir(){ echo $(local-base)/env/git/git-libgit2 ; }
libgit2-cd(){  cd $(libgit2-dir); }
libgit2-mate(){ mate $(libgit2-dir) ; }
libgit2-get(){
   local dir=$(dirname $(libgit2-dir)) &&  mkdir -p $dir && cd $dir

}
