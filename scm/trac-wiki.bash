
#
#  backup and restore of wiki pages via xmlrpc 
#
#    issues :
#         somehow getting permission error ... the log/trac.log becomes owned by root UNCONFIRMED 
#

trac-wiki-backup(){
  
  local name=${1:-$SCM_TRAC}
  shift
  
  cd $SCM_FOLD
  [ -d backup ] || ( sudo mkdir backup && sudo chown $USER backup )
  
  dir="backup/tracwikis/$SCM_HOST/$name/wiki"
  [ -d $dir ] || ( mkdir -p $dir || ( echo abort && return 1 ))
  cd $dir  
  python $HOME/$SCM_BASE/xmlrpc-wiki-backup.py $name $*
  
  ls -alst $dir
}

trac-wiki-restore(){
  
  local name=${1:-$SCM_TRAC}
  shift
  
  cd $SCM_FOLD
  dir="backup/tracwikis/$SCM_HOST/$name/wiki"
  [ -d $dir ] || ( echo abort ... must backup before can restore  && return 1 )
  cd $dir
  python $HOME/$SCM_BASE/xmlrpc-wiki-restore.py $name $*
}


trac-wiki-ls(){

  local name=${1:-$SCM_TRAC}
  shift
  
  cd $SCM_FOLD
  dir="backup/tracs/$SCM_HOST/$name/wiki"
    
   ls -alst $dir

}


