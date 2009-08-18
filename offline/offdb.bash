# === func-gen- : offline/offdb.bash fgp offline/offdb.bash fgn offdb
offdb-src(){      echo offline/offdb.bash ; }
offdb-source(){   echo ${BASH_SOURCE:-$(env-home)/$(offdb-src)} ; }
offdb-vi(){       vi $(offdb-source) ; }
offdb-env(){      elocal- ; }
offdb-usage(){
  cat << EOU
     offdb-src : $(offdb-src)

    Secure the mysqldump from Cheng Ju  
         env-htdocs-up ~/Downloads/$(offdb-name) 
         ( cd /tmp ; curl -O $(offdb-url))
         diff /tmp/$(offdb-name) ~/Downloads/$(offdb-name)  

    $(env-wikiurl)/OfflineDB




    offdb-build
        do the below 
 
 
    offdb-get
         get the dump 
    offdb-load 
         load into db 


EOU
}

offdb-dir(){ echo $(local-base)/env/offdb ; }
offdb-cd(){  cd $(offdb-dir) ; }
offdb-name(){ echo database.tar.gz ; }
offdb-url(){  echo $(env-htdocs-url $(offdb-name)) ; } 

offdb-build(){
   offdb-get
   offdb-fix
   offdb-load
   offdb-check
}

offdb-get(){
    local msg="=== $FUNCNAME :"
    local iwd=$PWD
    local dir=$(offdb-dir) 
    mkdir -p $dir && cd $dir
    local name=$(offdb-name)
    [ ! -f "$name" ] && curl -O $(offdb-url) && tar zxvf $name   ## one step as exploding tarball
    cd $iwd
}

offdb-fix(){
    local msg="=== $FUNCNAME :"
    offdb-cd 
}




offdb-load(){
   dj-
   dj-mysql < $(offdb-dir)/SimPmtSpec_data.sql
}

offdb-check(){
   dj-
   echo "select * from SimPmtSpec    ; " |  dj-mysql
   echo "select * from SimPmtSpecVld ; " |  dj-mysql
}


offdb-vld-columns(){ cat << EOC
      TIMESTART, TIMEEND, SITEMASK, SIMMASK, SUBSITE, TASK, AGGREGATENO, VERSIONDATE, INSERTDATE
EOC
}
offdb-double-vld(){ $FUNCNAME- | dj-mysql ; }
offdb-double-vld-(){ cat << EOS
      insert into SimPmtSpecVld (SEQNO, $(offdb-vld-columns)) select 2, $(offdb-vld-columns) from SimPmtSpecVld ; 
EOS 
}


offdb-pay-columns(){ cat << EOC
     PMTSITE, PMTAD, PMTRING, PMTCOLUMN, PMTGAIN, PMTGFWHM, PMTTOFFSET, PMTTSPREAD, PMTEFFIC, PMTPREPULSE, PMTAFTERPULSE, PMTDARKRATE
EOC
}
offdb-double-pay(){ $FUNCNAME- | dj-mysql ; }
offdb-double-pay-(){ cat << EOS
    insert into SimPmtSpec (SEQNO, $(offdb-pay-columns)) select 2, $(offdb-pay-columns) from SimPmtSpec ; 
EOS
}


offdb-kludge(){
   offdb-double-vld
   offdb-double-pay
}


