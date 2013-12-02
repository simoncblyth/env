# === func-gen- : git/gitsrc fgp git/gitsrc.bash fgn gitsrc fgh git
gitsrc-src(){      echo git/gitsrc.bash ; }
gitsrc-source(){   echo ${BASH_SOURCE:-$(env-home)/$(gitsrc-src)} ; }
gitsrc-vi(){       vi $(gitsrc-source) ; }
gitsrc-env(){      elocal- ; PATH=$(gitsrc-prefix)/bin:$PATH ; }
gitsrc-usage(){ cat << EOU

Building Git from Source
===========================

Belle7 SL5.1 git is too old to work with bitbucket.

* http://code.google.com/p/git-core/downloads/list

::

    [blyth@belle7 git-1.8.5]$ vi Makefile 
    [blyth@belle7 git-1.8.5]$ vi INSTALL 
    [blyth@belle7 git-1.8.5]$ make configure 
    GIT_VERSION = 1.8.5
        GEN configure
    [blyth@belle7 git-1.8.5]$ vi INSTALL 
    [blyth@belle7 git-1.8.5]$ ./configure --prefix=$(local-base)/env/gitsrc 
    [blyth@belle7 git-1.8.5]$ make all doc
        * new build flags
        CC credential-store.o
        * new link flags
        CC abspath.o

        GEN git-remote-testgit
    make -C Documentation all
    make[1]: Entering directory `/data1/env/local/env/git/git-1.8.5/Documentation'
        GEN mergetools-list.made
        GEN cmd-list.made
        GEN doc.dep
    make[2]: Entering directory `/data1/env/local/env/git/git-1.8.5'
    make[2]: `GIT-VERSION-FILE' is up to date.
    make[2]: Leaving directory `/data1/env/local/env/git/git-1.8.5'
    make[1]: Leaving directory `/data1/env/local/env/git/git-1.8.5/Documentation'
    make[1]: Entering directory `/data1/env/local/env/git/git-1.8.5/Documentation'
    make[2]: Entering directory `/data1/env/local/env/git/git-1.8.5'
    make[2]: `GIT-VERSION-FILE' is up to date.
    make[2]: Leaving directory `/data1/env/local/env/git/git-1.8.5'
        ASCIIDOC git-add.html
    /bin/sh: line 1: asciidoc: command not found
    make[1]: *** [git-add.html] Error 127
    make[1]: Leaving directory `/data1/env/local/env/git/git-1.8.5/Documentation'
    make: *** [doc] Error 2
    [blyth@belle7 git-1.8.5]$ 


    [blyth@belle7 git-1.8.5]$ sudo make install



* http://www.methods.co.nz/asciidoc/




EOU
}
gitsrc-name(){ echo git-1.8.5 ; }
gitsrc-dir(){ echo $(local-base)/env/git/$(gitsrc-name) ; }
gitsrc-cd(){  cd $(gitsrc-dir); }
gitsrc-mate(){ mate $(gitsrc-dir) ; }
gitsrc-get(){
   local dir=$(dirname $(gitsrc-dir)) &&  mkdir -p $dir && cd $dir

   local url=http://git-core.googlecode.com/files/$(gitsrc-name).tar.gz
   local tgz=$(basename $url)
   local nam=${tgz/.tar.gz}
   [ ! -f "$tgz" ] && curl -L -O $url
   [ ! -d "$nam" ] && tar zxvf $tgz

}

gitsrc-prefix(){
    echo $(local-base)/env/gitsrc
}
