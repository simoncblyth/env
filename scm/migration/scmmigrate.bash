# === func-gen- : scm/migration/scmmigrate fgp scm/migration/scmmigrate.bash fgn scmmigrate fgh scm/migration
scmmigrate-src(){      echo scm/migration/scmmigrate.bash ; }
scmmigrate-source(){   echo ${BASH_SOURCE:-$(env-home)/$(scmmigrate-src)} ; }
scmmigrate-vi(){       vi $(scmmigrate-source) ; }
scmmigrate-env(){      elocal- ; }
scmmigrate-usage(){ cat << EOU

SCM MIGRATE
============

Uses standard scm repos/tracs/backup layout adopted 
by all my SVN/Trac instances to facilitate migrations.


FUNCTIONS
----------

*scmmigrate-get*
    transfers backup tarballs from remote hub node, verifies and unpacks them  
    into same directory layout as source, namely beneath /var/scm/backup


*scmmigrate-tickets*
    



EOU
}
scmmigrate-mate(){ mate $(scmmigrate-dir) ; }
scmmigrate-hub(){ echo ${SCMMIGRATE_HUB:-C2} ; }
scmmigrate-repo(){ echo ${SCMMIGRATE_REPO:-env} ; }
scmmigrate-stamp(){ echo ${SCMMIGRATE_STAMP:-2014/07/20/173006} ; } # could be "last" 
scmmigrate-fold(){ echo /var/scm ; }

scmmigrate-dump(){ cat << EOI

  scmmigrate-hub          :  $(scmmigrate-hub)
  scmmigrate-repo         :  $(scmmigrate-repo)
  scmmigrate-stamp        :  $(scmmigrate-stamp)

  *scmmigrate-get* copies remote hub node directories to local, 
  does dna check and untars the tarball 

  scmmigrate-tgzdir tracs :  $(scmmigrate-tgzdir tracs)
  scmmigrate-tgzdir repos :  $(scmmigrate-tgzdir repos)


  scmmigrate-tracdir      : $(scmmigrate-tracdir)
  scmmigrate-svndir       : $(scmmigrate-svndir)

EOI
}


scmmigrate-tgzdir(){
   local typ=${1:-tracs}
   local stamp=$(scmmigrate-stamp)
   local repo=$(scmmigrate-repo)
   local fold=$(scmmigrate-fold)
   case $typ in 
      tracs) echo $fold/backup/$(local-tag2node $(scmmigrate-hub))/tracs/${repo}/${stamp} ;;
      repos) echo $fold/backup/$(local-tag2node $(scmmigrate-hub))/repos/${repo}/${stamp} ;;
   esac
}

scmmigrate-tracdir-cd(){ cd $(scmmigrate-tracdir); }
scmmigrate-tracdir(){
   local tgzdir=$(scmmigrate-tgzdir tracs)
   local nam=$(scmmigrate-repodirname $tgzdir)
   echo $tgzdir/$nam
}

scmmigrate-svndir-cd(){ cd $(scmmigrate-svndir); }
scmmigrate-svndir(){
   local tgzdir=$(scmmigrate-tgzdir repos)
   local nam=$(scmmigrate-repodirname $tgzdir)
   echo $tgzdir/$nam
}

scmmigrate-export(){
   export SCMMIGRATE_TRACDIR=$(scmmigrate-tracdir)
   export SCMMIGRATE_SVNDIR=$(scmmigrate-svndir)
   scmmigrate-info
}
scmmigrate-info(){
  env | grep SCMMIGRATE 
}

scmmigrate-cd(){
   local dir=$(scmmigrate-dir)
   cd $dir
}

scmmigrate-get(){
   local repotypes="tracs repos"
   local repotype
   local tgzdir
   for repotype in $repotypes ; do 
       local tgzdir=$(scmmigrate-tgzdir $repotype)
       scmmigrate-get-tgzdir $tgzdir
       scmmigrate-untar-tgzdir $tgzdir
   done
}

scmmigrate-get-tgzdir(){
   local msg="=== $FUNCNAME : "
   local tgzdir=$1
   local hub=$(scmmigrate-hub)
   local nam=$(basename $tgzdir)
   local dir=$(dirname $tgzdir)
   [ ! -d "$dir" ] && echo $msg creating dir $dir && mkdir -p $dir  
   cd $dir
   [ ! -d "${nam}" ]     && echo $msg scp -r $tgzdir  && scp -r $hub:${tgzdir} .
   [ -d "${nam}" ]       && echo $msg tgzdir $tgzdir exists  
}


scmmigrate-repodirname(){
   local dir=${1:-.}

   [ ! -d "$dir" ] && echo repodir-does-not-exist-yet && return 

   local tgz=$(ls -1 $dir/*.tar.gz)
   local bas=$(basename $tgz)
   local nam=${bas/.tar.gz}
   echo $nam
}

scmmigrate-untar-tgzdir(){
   local msg="=== $FUNCNAME :"
   local dir=${1:-.}
   local tgz=$(ls -1 $dir/*.tar.gz)
   local nam=$(scmmigrate-repodirname $dir)
   local iwd=$PWD
   cd $dir
   local rc
   dna.py $tgz
   rc=$?

   [ ${rc} -ne 0 ] && echo $msg WARNING DNA MISMATCH && sleep 1000000000000
   [ ! -d "$nam" ] && echo $msg untarring $tgz && tar zxf $tgz
   [ -d "$nam" ] && echo $msg tarballs from $dir are untarred into $nam

   cd $iwd
}


scmmigrate-tickets(){
   local repodir=$(scmmigrate-tracdir)
   echo $msg repodir $repodir 
   trac2bitbucket-
   trac2bitbucket-tickets $repodir 
}
