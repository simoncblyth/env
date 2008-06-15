
_annobit-usage(){

cat << EOU

  Exploring the SQL needed to access the annotation info needed
  from the bitten_report* tables

   _annobit-getdb  :
       get a copy of the trac datbase from the last backup tarball  

   _annobit-sql    :
       run some short quoted sql against the db,  eg:  
           _annobit-sql  "select * from bitten_report ; " 
           _annobit-sql  "$(cat q.sql)"
 
 
    _annobit-sql-  :
        unquoted invokation for piping in a file of commands
          _annobit-sql- < q.sql
 
          _annobit-sql- 
 
 
    _annobit-boost-bitten :
         establish the correspondence between boost-bitten and bittem
 
 
 
 
EOU

}


_annobit-env(){
  elocal-
  export ANNOBIT_OPT="-column -header"
}


_annobit-opt(){
  echo $ANNOBIT_OPT
}

_annobit-db(){
  echo /tmp/env/annobit-getdb/workflow/db/trac.db
}

_annobit-getdb(){

  local tmp=/tmp/env/$FUNCNAME && mkdir -p $tmp
  local iwd=$PWD

  cd $tmp

  local db=workflow/db/trac.db
  [ ! -f $db ] && tar zxvf $SCM_FOLD/backup/g4pb/tracs/workflow/last/workflow.tar.gz $db
  ls -l $db
  
  #cd $iwd

}

_annobit-sql-(){
  sqlite3 $(_annobit-opt)  $(_annobit-db) $*
}

_annobit-sql(){
  sqlite3 $(_annobit-opt) $(_annobit-db)  "$*" 
}

_annobit-sql-demo(){

  _annobit-sql-  << EOD
  
select * from bitten_report ;
select * from bitten_report_item ;
select * from bitten_report ;
  
EOD

}

_annobit-sql-hmm(){


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

  _annobit-sql- < $tmp



}


_annobit-boost-bitten(){

  

  local tmp=/tmp/env/$FUNCNAME && mkdir -p $tmp
  
  cd $tmp
  svn co https://svn.boost-consulting.com/boost/bitten/  boost-bitten
  svn co http://svn.edgewall.org/repos/bitten/trunk bitten-trunk
  
  
  svn co -r 2654 https://svn.boost-consulting.com/boost/bitten/  boost-bitten-r2654
  svn co -r 516 http://svn.edgewall.org/repos/bitten/trunk bitten-trunk-r516 
   
  diff -r --brief boost-bitten bitten-trunk | grep -v .svn 
   
  diff -r --brief boost-bitten bitten-trunk | grep -v .svn  | grep Files | perl -p -e 's/Files (\S*) and (\S*) differ/opendiff $1 $2/' - 
   

#opendiff boost-bitten/bitten/build/ctools.py bitten-trunk/bitten/build/ctools.py            ## trunk autoconf addition
#opendiff boost-bitten/bitten/build/pythontools.py bitten-trunk/bitten/build/pythontools.py  ## trunk figleaf addition
#opendiff boost-bitten/bitten/build/tests/pythontools.py bitten-trunk/bitten/build/tests/pythontools.py  ## figleaf
#opendiff boost-bitten/bitten/build/shtools.py bitten-trunk/bitten/build/shtools.py          ## dir_ addition to exec
#opendiff boost-bitten/bitten/build/tests/api.py bitten-trunk/bitten/build/tests/api.py       ## minor trunk fixes
#opendiff boost-bitten/bitten/master.py bitten-trunk/bitten/master.py         ## minor trunk fixes
#opendiff boost-bitten/bitten/report/coverage.py bitten-trunk/bitten/report/coverage.py   ## trunk addition of 0.11 testcoverage annotator
#opendiff boost-bitten/bitten/report/tests/coverage.py bitten-trunk/bitten/report/tests/coverage.py     ## truck stub set up enhancement
#opendiff boost-bitten/bitten/slave.py bitten-trunk/bitten/slave.py      ## trunk minor fixes
#opendiff boost-bitten/bitten/tests/admin.py bitten-trunk/bitten/tests/admin.py   ## trunk permission fixes
#opendiff boost-bitten/bitten/tests/master.py bitten-trunk/bitten/tests/master.py   ## trunk wrong slave enhancement
#opendiff boost-bitten/bitten/util/testrunner.py bitten-trunk/bitten/util/testrunner.py  ## trunk figleaf addition and filtering 
#opendiff boost-bitten/doc/commands.txt bitten-trunk/doc/commands.txt  ## this is generated documentation 
#opendiff boost-bitten/setup.py bitten-trunk/setup.py

#opendiff boost-bitten/bitten/queue.py bitten-trunk/bitten/queue.py           ## build deletion ... some divergence  ... looks like that patch

#opendiff boost-bitten/bitten/templates/bitten_build.cs bitten-trunk/bitten/templates/bitten_build.cs   ## *** boost addition .. all_steps checkbox
#opendiff boost-bitten/bitten/templates/bitten_config.cs bitten-trunk/bitten/templates/bitten_config.cs  ## *** boost addition ... build progress
#opendiff boost-bitten/bitten/templates/bitten_summary_tests.cs bitten-trunk/bitten/templates/bitten_summary_tests.cs  ## *** boost addition ... view output_href
#opendiff boost-bitten/bitten/htdocs/bitten.css bitten-trunk/bitten/htdocs/bitten.css      ## ***boost addition ... progress table
#opendiff boost-bitten/bitten/report/testing.py bitten-trunk/bitten/report/testing.py     ## *** boost addition ... stdout in tables and DetailsComponent 
#opendiff boost-bitten/bitten/web_ui.py bitten-trunk/bitten/web_ui.py   ## **** boost additions

#
#Only in bitten-trunk/bitten/htdocs: bitten_coverage.css
#Only in boost-bitten/bitten/templates: bitten_test_details.html
#

}






  



