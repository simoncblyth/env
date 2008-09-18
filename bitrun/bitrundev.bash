

bitrundev-usage(){

  cat << EOU
  
   Extraction of non-portable bits of bitrun.bash not needed for standard running


     bitrundev-hotcopy <name> :
           make a hotcopy of the trac database, allowing safe sqlite access                      
                                                
     bitrundev-sqlite 
           connect to the hotcopied database
           
     bitrundev-fluff
           touch and check in to force slave run 
           
     bitrundev-path  : $(bitrundev-path)
           development only convenience
           its better for the slave not to need to know this 
 
           

EOU

}


bitrundev-env(){
  bitrun-
}

bitrundev-hotcopy(){
   local name=$(bitrun-instance)
   shift
   local tmp=/tmp/$name/${FUNCNAME/-*/} && mkdir -p $tmp   
   local cmd="cd $tmp && rm -rf hotcopy && sudo trac-admin $SCM_FOLD/tracs/$name hotcopy hotcopy && sudo chown -R $USER hotcopy"
   
   echo $cmd
   eval $cmd 
   
}

bitrundev-sqlite(){
   local name=$(bitrun-instance)
   local tmp=/tmp/$name/${FUNCNAME/-*/} && mkdir -p $tmp 
   cd $tmp
   
   sqlite3 hotcopy/db/trac.db

}

bitrundev-fluff(){
    local msg="=== $FUNCNAME: $* "
    local fluff=$WORKFLOW_HOME/demo/fluff.txt
    date >> $fluff
    local cmd="svn ci $fluff -m \"$msg\" "
    echo $cmd
    eval $cmd
}

bitrundev-path(){

   local name=$(bitrun-instance)
   case $name in
     workflow) echo trunk/demo ;;
          env) echo trunk/unittest/demo ;;
            *) echo error-$FUNCNAME ;;
   esac
}






