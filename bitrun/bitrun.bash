
bitrun-usage(){

cat << EOU

    bitrun-url   : $(bitrun-url)
    bitrun-cfg   : $(bitrun-cfg)
    bitrun-path  : $(bitrun-path)
         its better for the slave not to need to know this 

    BITRUN_OPT   : $BITRUN_OPT
    TRAC_INSTANCE : $TRAC_INSTANCE

    which bitten-slave : $(which bitten-slave)


    bitrun-dumb    :   pure default running 
            
    bitrun-start :
     
       the build-dir is by default created within the work-dir with a name like 
       build_${config}_${build}   setting it to "" as used here  is a convenience for testing
       which MUST go together with "--keep-files" to avoid potentially deleting bits 
       of working copy      
    


    recipe tips
       -  shield the slave from non-zero return codes with "echo $?" for example 
       -  escaping backslashes in xml is problematic, why ?
              when doing ${p/trunk\//} it somehow becomes  ${p/trunk\\//} which doesnt work
              avoid the issue by using   .${p/trunk/} 
       - be aware of the different directories in use
             - the invokation context in /tmp/env/bitrun-start/etc..
             - it seems that sh:exec gets along fine with that ... or pehaps its just capturing stdout and ignoring 
               the files it is creating ???
               ... but the explicit report needs a path 
               
               Failed to read test report at /private/tmp/env/bitrun-start/nosetests.xml          
        

     bitrun-hotcopy <name> :
           make a hotcopy of the trac database                        
                                                
     bitrun-sqlite 
           connect to the hotcopied database

     the new stuff is getting into the database...  but not appearing in annotation 
           

1|10|type|test
1|11|status|failure
1|11|name|test_odds(5, 15)
1|11|stdout|Traceback (most recent call last):

  File "/System/Library/Frameworks/Python.framework/Versions/2.5/lib/python2.5/unittest.py", line 260, in run
    testMethod()

  File "/Library/Python/2.5/site-packages/nose-0.10.3-py2.5.egg/nose/case.py", line 182, in runTest
    self.test(*self.arg)

  File "/Users/blyth/workflow/demo/package/module_test.py", line 60, in check_even
    assert n % 2 == 0 or nn % 2 == 0

AssertionError

1|11|lines|60
1|11|fixture|package.module_test.test_odds(5, 15)
1|11|range|52-56
1|11|file|package/module_test.py
1|11|duration|0.000236
1|11|type|test
1|12|failures|3
1|12|tests|12                
                                                         
                                                                                                       
                                                                                                                                               

EOU
}


bitrun-hotcopy(){
   local name=${1:-$TRAC_INSTANCE}
   shift
   local tmp=/tmp/env/${FUNCNAME/-*/} && mkdir -p $tmp   
   local cmd="cd $tmp && rm -rf hotcopy && sudo trac-admin $SCM_FOLD/tracs/$name hotcopy hotcopy && sudo chown -R $USER hotcopy"
   
   echo $cmd
   eval $cmd 
   
}

bitrun-sqlite(){
   local tmp=/tmp/env/${FUNCNAME/-*/} && mkdir -p $tmp 
   cd $tmp
   
   sqlite3 hotcopy/db/trac.db

}


bitrun-env(){
  elocal-
  trac-    ## pick up TRAC_INSTANCE for the node 
  export BITRUN_OPT="--dry-run"
}

bitrun-url(){
   local url
   
   local name=${1:-workflow}
   case $name in
     workflow) url=http://localhost/tracs/$name/builds ;;
          env) url=http://dayabay.phys.ntu.edu.tw/tracs/$name/builds ;;
            *) url=
   esac
   echo $url
}

bitrun-path(){
   case ${1:-workflow} in
     workflow) echo trunk/demo ;;
          env) echo trunk/unittest/demo ;;
            *) echo error-$FUNCNAME ;;
   esac
}

bitrun-cfg(){
    echo $ENV_HOME/bitrun/$LOCAL_NODE.cfg
}

bitrun-fluff(){
    local msg="=== $FUNCNAME: $* "
    local fluff=$WORKFLOW_HOME/demo/fluff.txt
    date >> $fluff
    local cmd="svn ci $fluff -m \"$msg\" "
    echo $cmd
    eval $cmd
}







bitrun-dumb(){
   bitten-slave $(bitrun-url $*)
}



bitrun-cmd-(){
   local name=$1
   shift
   local cmd=$(cat << EOC
      bitten-slave -v -f $(bitrun-cfg) --dump-reports -u blyth -p $NON_SECURE_PASS $* $(bitrun-url $name)
EOC)
    echo $cmd
}


bitrun-start(){

    local name=${1:-$TRAC_INSTANCE}
    shift

    local iwd=$PWD
    local msg="=== $FUNCNAME :"

    [ ! -f $cfg ] && echo $msg ERROR no bitten config file $file for LOCAL_NODE $LOCAL_NODE && return 1

    local tmp=/tmp/env/${FUNCNAME/-*/} && mkdir -p $tmp
    cd $tmp
    
    local cmd=$(bitrun-cmd- $name  --work-dir=. --build-dir=  --keep-files) 
    echo $cmd
    eval $cmd
  
    cd $iwd
}









