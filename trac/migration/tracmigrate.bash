# === func-gen- : trac/migration/tracmigrate fgp trac/migration/tracmigrate.bash fgn tracmigrate fgh trac/migration
tracmigrate-src(){      echo trac/migration/tracmigrate.bash ; }
tracmigrate-source(){   echo ${BASH_SOURCE:-$(env-home)/$(tracmigrate-src)} ; }
tracmigrate-vi(){       vi $(tracmigrate-source) ; }
tracmigrate-env(){      elocal- ; }
tracmigrate-usage(){ cat << EOU

TRACMIGRATE
===========

*tracmigrate-get*

     #. copies trac repo tarballs and sidecar dna over from hub node
     #. checks dna match 
     #. untars the tarball

NEXT

#. generalize to scm-migrate, need SVN repo 
   for HG vs SVN history comparisons


EOU
}
tracmigrate-mate(){ mate $(tracmigrate-dir) ; }

tracmigrate-hub(){ echo C2 ; }
tracmigrate-repo(){ echo ${TRACMIGRATE_REPO:-env} ; }
tracmigrate-tgz(){
   local stamp=2014/07/20/173006   # could be "last"
   local hub=$(tracmigrate-hub)
   local repo=$(tracmigrate-repo)
   local tgz=/var/scm/backup/$(local-tag2node $(tracmigrate-hub))/tracs/${repo}/${stamp}/${repo}.tar.gz 
   echo $tgz
}
tracmigrate-dir(){
   local tgz=$(tracmigrate-tgz)
   echo $(dirname $tgz)
}
tracmigrate-cd(){
   local dir=$(tracmigrate-dir)
   cd $dir
}
tracmigrate-get(){
   local msg="=== $FUNCNAME : "
   local tgz=$(tracmigrate-tgz)
   local nam=$(basename $tgz)
   nam=${nam/.tar.gz}
   local hub=$(tracmigrate-hub)
   local dir=$(dirname $tgz)
   [ ! -d "$dir" ] && echo $msg creating dir $dir && mkdir -p $dir  
   cd $dir

   [ ! -f "${tgz}" ]     && echo $msg scp $tgz     && scp $hub:${tgz} .
   [ ! -f "${tgz}.dna" ] && echo $msg scp $tgz.dna && scp $hub:${tgz}.dna .

   local rc
   dna.py $tgz
   rc=$?

   [ ${rc} -ne 0 ] && echo $msg WARNING DNA MISMATCH && sleep 1000000000000
   [ ! -d "$nam" ] && echo $msg untarring $tgz && tar zxvf $tgz
}

tracmigrate-repodir(){ echo $(tracmigrate-dir)/$(tracmigrate-repo) ; }


tracmigrate-tickets(){
   local repodir=$(tracmigrate-repodir)
   echo $msg repodir $repodir 
   trac2bitbucket-
   trac2bitbucket-tickets $repodir 

}


