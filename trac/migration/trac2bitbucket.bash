# === func-gen- : trac/migration/trac2bitbucket fgp trac/migration/trac2bitbucket.bash fgn trac2bitbucket fgh trac/migration
trac2bitbucket-src(){      echo trac/migration/trac2bitbucket.bash ; }
trac2bitbucket-source(){   echo ${BASH_SOURCE:-$(env-home)/$(trac2bitbucket-src)} ; }
trac2bitbucket-vi(){       vi $(trac2bitbucket-source) ; }
trac2bitbucket-env(){      elocal- ; }
trac2bitbucket-usage(){ cat << EOU

Trac2Bitbucket 
==================

Bitbucket documents an issue json format, that looks like
a good format to use to hold issue info in flexible manner.

* https://confluence.atlassian.com/pages/viewpage.action?pageId=330796872


EOU
}
trac2bitbucket-dir(){ echo $(local-base)/env/trac/migration/trac2bitbucket ; }
trac2bitbucket-cd(){  cd $(trac2bitbucket-dir); }
trac2bitbucket-mate(){ mate $(trac2bitbucket-dir) ; }
trac2bitbucket-get(){
   local dir=$(dirname $(trac2bitbucket-dir)) &&  mkdir -p $dir && cd $dir

   hg clone https://bitbucket.org/secdev/trac2bitbucket

}



