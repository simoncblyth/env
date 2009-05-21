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


   Observations in dybgaudi/Database

     * ENV_TSQL_* ... cascades handled by colons 
       ./DatabaseInterface/src/DbiCascader.cxx

     * dybgaudi/Database/DatabaseMaintenance/tools 
            all appear to be wrappers around "mysql" and "mysqldump" 
            no other means of access (such as the standard perl DBI)

     * DBI compliant "name" table 
         * has companion "nameVLD" table which has SEQNO
         * SEQNO, ROW_COUNTER columns 

     * dybgaudi/Database/DatabaseInterface/DatabaseInterface/DbiResultPtr.h
            the primary way in 

     * dybgaudi/Database/DbiDataSvc/src/components/DbiSimDataSvc.cc
           demo of the usage ...


{{{
  TimeStamp tstamp(2007, 1, 15, 0, 0, 1);
  Context vc(Site::kDayaBay, SimFlag::kMC, tstamp);
  DbiResultPtr<SimPmtSpec> pr;
  pr.NewQuery(vc);

  // Delete any residual objects and check for leaks.

  DbiTableProxyRegistry::Instance().ShowStatistics();
  

  // Check number of entires in result set
  unsigned int numRows = pr.GetNumRows();
  std::cout << "CJSLIN: Database rows = " << numRows << std::endl;
  const SimPmtSpec* row ;
  int site, detectorID, ring, column;
}}}


      TASK ... LOCATE THE SQL QUERY THAT WAS DONE FOR THIS "DBI" QUERY ...


   Exposing the functionality of dybgaudi/Database/DatabaseInterface 
   to python, especially query preparation in 
        dybgaudi/Database/DatabaseInterface/src/DbiDBProxy.cxx 
   without using Gaudi  ... will allow a django webapp to 
   make DBI like queries ?

  



EOU
}

offdb-dir(){ echo $(local-base)/env/offdb ; }
offdb-name(){ echo database.tar.gz ; }
offdb-url(){  echo $(env-htdocs-url $(offdb-name)) ; } 
offdb-get(){

    local msg="=== $FUNCNAME :"
    local iwd=$PWD
    local dir=$(offdb-dir) 
    mkdir -p $dir && cd $dir

    local name=$(offdb-name)
    [ ! -f "$name" ] && curl -O $(offdb-url) && tar zxvf $name   ## one step as exploding tarball



    #cd $iwd
}



