# === func-gen- : aberdeen/runinfo/runinfo fgp aberdeen/runinfo/runinfo.bash fgn runinfo fgh aberdeen/runinfo
runinfo-src(){      echo aberdeen/runinfo/runinfo.bash ; }
runinfo-source(){   echo ${BASH_SOURCE:-$(env-home)/$(runinfo-src)} ; }
runinfo-vi(){       vi $(runinfo-source) ; }
runinfo-env(){      
   elocal- 

   export DJANGO_PROJECT=runinfo
   export DJANGO_APP=run
   export DJANGO_PROJDIR=$(runinfo-dir)
   dj-
}
runinfo-usage(){
  cat << EOU


     NB following runinfo- precursor the DJANGO_* "context" is switched to runinfo 
        providing generic django functionality via the dj- and djdep- funcs
 

     runinfo-dir : $(runinfo-dir)

     runinfo-build  : 
          specialization of dj-build with the runinfo-env

     runinfo-ingest
          ingest from Jimmys MIDAS runlog csv list 

     runinfo-sv
          add the runinfo app to supervisor (sv-) control ready 
          for non-embedded deployment

     runinfo-celeryd
          interactive run of the celery daemon (for testing), 
          the daemon makes periodic checks of the message queue and 
          takes required actions when new messages are found, such 
          as ingesting serialized objects into the database

          see tasks.py and messaging.py for implementation 

EOU
}
runinfo-dir(){ echo $(env-home)/aberdeen/runinfo ; }
runinfo-cd(){  cd $(runinfo-dir); }
runinfo-mate(){ mate $(runinfo-dir) ; }

runinfo-build(){  dj-build ; }
runinfo-ingest(){ dj-manage csv_ingest ; }
runinfo-sv(){     djdep-;djdep-sv ; }


runinfo-celeryd(){  dj-manage celeryd ; } 


