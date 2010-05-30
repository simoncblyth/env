# === func-gen- : cjsn/cjsn fgp cjsn/cjsn.bash fgn cjsn fgh cjsn
cjsn-src(){      echo cjsn/cjsn.bash ; }
cjsn-source(){   echo ${BASH_SOURCE:-$(env-home)/$(cjsn-src)} ; }
cjsn-vi(){       vi $(cjsn-source) ; }
cjsn-env(){      
   elocal- 
   export CJSN_HOME=$(cjsn-home)
}
cjsn-usage(){
  cat << EOU
     cjsn-src : $(cjsn-src)
     cjsn-dir : $(cjsn-dir)

     An earlier Makefile based approach is in cjson-

     This is much simpler as : 
        * using SCT/SCons for building so management of 
          derived files is automatic

     cjsn-url : $(cjsn-url)
     cjsn-rev : $(cjsn-rev)
     cjsn-get
         exports the pinned revision $(cjsn-rev) from the upstream repo

  == Issues ==

    [blyth@cms02 e]$ cjsn-get
    svn: SSL is not supported

    Workaround : use a different svn :
      SVN=/usr/bin/svn cjsn-get


EOU
}
cjsn-home(){ echo $(local-base)/env/cjsn ; }
cjsn-cd(){  cd $(cjsn-home); }
cjsn-mate(){ mate $(cjsn-home) ; }

cjsn-rev(){  echo 33 ; }
cjsn-url(){  echo https://cjson.svn.sourceforge.net/svnroot/cjson@$(cjsn-rev) ; }
cjsn-get(){
   local iwd=$PWD
   local dir=$(dirname $(cjsn-home)) &&  mkdir -p $dir && cd $dir
   ${SVN:-svn} co $(cjsn-url) cjsn
   cd $iwd
}
cjsn-build(){
   sct-
   sct cjsn
}


