
annobit-usage(){

cat << EOU

  Exploring the SQL needed to access the annotation info needed
  from the bitten_report* tables

   annobit-getdb  :
       get a copy of the trac datbase from the last backup tarball  

   annobit-sql    :
       run some short quoted sql against the db,  eg:  
           annobit-sql  "select * from bitten_report ; " 
           annobit-sql  "$(cat q.sql)"
 
 
    annobit-sql-  :
        unquoted invokation for piping in a file of commands
          annobit-sql- < q.sql
 
          annobit-sql- 
 
 
 
EOU

}


annobit-env(){
  elocal-
  export ANNOBIT_OPT="-column -header"
}


annobit-opt(){
  echo $ANNOBIT_OPT
}

annobit-db(){
  echo /tmp/env/annobit-getdb/workflow/db/trac.db
}

annobit-getdb(){

  local tmp=/tmp/env/$FUNCNAME && mkdir -p $tmp
  local iwd=$PWD

  cd $tmp

  local db=workflow/db/trac.db
  [ ! -f $db ] && tar zxvf $SCM_FOLD/backup/g4pb/tracs/workflow/last/workflow.tar.gz $db
  ls -l $db
  
  #cd $iwd

}

annobit-sql-(){
  sqlite3 $(annobit-opt)  $(annobit-db) $*
}

annobit-sql(){
  sqlite3 $(annobit-opt) $(annobit-db)  "$*" 
}

annobit-sql-demo(){

  annobit-sql-  << EOD
  
select * from bitten_report ;
select * from bitten_report_item ;
select * from bitten_report ;
  
EOD

}

annobit-sql-hmm(){


#
#
#
#  bitten_report
#    id          build       step        category    generator 
#                             xmltest
#
#
#
#  bitten_report_item
#
#      report item name value  
#
#      id=1    0..  file     
#                   fixture
#
#
#
#
#


  local build=1
  local step=xmltest
  local tmp=/tmp/env/$FUNCNAME.sql
  local status=failure

  cat << EOQ > $tmp
SELECT build.config, build.rev, config.path, item_file.report,item_file.item,item_file.value,item_status.value,item_fixture.value  \
 FROM bitten_report AS report \
 LEFT OUTER JOIN bitten_build  AS build  ON report.build=build.id \
 LEFT OUTER JOIN bitten_config AS config ON build.config=config.name \
 LEFT OUTER JOIN bitten_report_item AS item_name    ON (   item_name.report=report.id AND item_name.name='name') \
 LEFT OUTER JOIN bitten_report_item AS item_fixture ON (item_fixture.report=report.id AND item_fixture.item=item_name.item AND item_fixture.name='fixture') \
 LEFT OUTER JOIN bitten_report_item AS item_status  ON ( item_status.report=report.id AND item_status.item=item_name.item AND item_status.name='status') \
 LEFT OUTER JOIN bitten_report_item AS item_file    ON (   item_file.report=report.id AND item_file.item=item_name.item   AND item_file.name='file') \
WHERE build=$build AND step='$step'  ;

EOQ

  cat $tmp

  annobit-sql- < $tmp



}



  



