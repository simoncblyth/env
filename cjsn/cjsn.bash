# === func-gen- : cjsn/cjsn fgp cjsn/cjsn.bash fgn cjsn fgh cjsn
cjsn-src(){      echo cjsn/cjsn.bash ; }
cjsn-source(){   echo ${BASH_SOURCE:-$(env-home)/$(cjsn-src)} ; }
cjsn-vi(){       vi $(cjsn-source) ; }
cjsn-env(){      elocal- ; }
cjsn-usage(){
  cat << EOU
     cjsn-src : $(cjsn-src)
     cjsn-dir : $(cjsn-dir)

     An earlier Makefile based approach is in cjson-

     This is much simpler as : 
        * using SCT/SCons for building so management of 
          of derived files is automatic

     cjsn-url : $(cjsn-url)
     cjsn-rev : $(cjsn-rev)
     cjsn-get
         exports the pinned revision $(cjsn-rev) from the upstream repo

EOU
}
cjsn-dir(){ echo $(env-home)/cjsn/src ; }
cjsn-cd(){  cd $(cjsn-dir); }
cjsn-mate(){ mate $(cjsn-dir) ; }

cjsn-rev(){  echo 33 ; }
cjsn-url(){  echo https://cjson.svn.sourceforge.net/svnroot/cjson@$(cjsn-rev) ; }
cjsn-get(){
   local dir=$(dirname $(cjsn-dir)) &&  mkdir -p $dir && cd $dir
   svn export $(cjsn-url)  src
}
cjsn-build(){
   sct-
   sct cjsn
}


