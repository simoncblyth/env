     
 dyw_iwd=$(pwd)          
                       
 DYW_BASE=$ENV_BASE/dyw
 export DYW_HOME=$HOME/$DYW_BASE

 cd $DYW_HOME

 [ -r geant4-env.bash ]  && . geant4-env.bash
 [ -r dyw_use.bash ]     && . dyw_use.bash 

 dyw-setup(){ 
     ## setup the paths etc.. 
     
 
     
    [ "$DYW_DBG" == "1" ] && echo ======= dyw-setup invoked 
    [ -r $DYW/G4dyb/cmt/setup.sh ] && . $DYW/G4dyb/cmt/setup.sh 
    [ "$DYW_DBG" == "1" ] && echo ======= dyw-setup completed
}

 if [ "$NODE_TAG" == "G" ]; then
    [  "$DYW_DBG" == "1" ] && echo skipping dyw-setup invokation on node $NODE_TAG
 elif [ "$NODE_TAG" == "P" ]; then
    #echo "skipping dyw-setup on node $NODE_TAG as attempting dyb installation "
    dyw-setup
 else  
    dyw-setup 
 fi
 
 [ -r dyw_gen.bash ] && . dyw_gen.bash
 
 
 ##  caution must exit with initial dir
 cd $dyw_iwd
 [ -t 0 ]   || return 
 [ "$TZERO_DBG" == "1" ]  && echo faked tzero  && return 
 cd $DYW_HOME
 
 [ -r dyw_build.bash ] && . dyw_build.bash
 
 ## hmmm, needs to come after she soxt.bash call ...
 soxt-env
 

 ##  caution must exit with initial dir
 cd $dyw_iwd

root-use(){ [ -r $DYW_HOME/root_use.bash ] && . $DYW_HOME/root_use.bash ; }
dyw(){     [ -r $DYW_HOME/dyw.bash ]     && . $DYW_HOME/dyw.bash ; }
dyw-osx(){ [ -r $DYW_HOME/dyw-osx.bash ] && . $DYW_HOME/dyw-osx.bash ; }


## useful when cannot login due to disk hangs  , flipping the return commenting 
dyw-x-off(){ ssh ${1:-$TARGET_TAG} "perl -pi -e 's/^#return/return/' $DYW_BASE/dyw.bash" ; }
dyw-x-on(){  ssh ${1:-$TARGET_TAG} "perl -pi -e 's/^return/#return/' $DYW_BASE/dyw.bash" ; }

## propagating the env to other machines
dyw-x-pkg(){ 
   cd $HOME 	
   tar zcvf dyw.tar.gz $DYW_BASE/*
   scp dyw.tar.gz ${1:-$TARGET_TAG}:; 
   ssh ${1:-$TARGET_TAG} "tar zxvf dyw.tar.gz" 
}

dyw-x(){ scp $HOME/$DYW_BASE/dyw.bash ${1:-$TARGET_TAG}:$DYW_BASE; }
dyw-i(){ .   $HOME/$DYW_BASE/dyw.bash ; }

dyw-vi(){
	  iwd=$(pwd)    
	  cd $HOME/$DYW_BASE 
	  vi *
	  cd $iwd
}


dyw-x-sync(){
   X=${1:-$TARGET_TAG}
   vname="DYW_$X"
   eval DYW_X=\$$vname

   if [ "X$DYW_X" == "X" ]; then
	  echo "to syncronise to node $X must set up the $vname variable in $DYW_BASE/dyw_use.bash " 
   else
      echo "syncronise $DYW local copy of cvs repository to $X:$vname ie $DYW_X  ... first a dry run.. then the real thing "
      [ "$LOCAL_NODE" == "$SOURCE_NODE" ] && (  ssh $X mkdir -p $DYW_X ) || echo "cannot create macro folder $DYW_X on node $X " 
      [ "$LOCAL_NODE" == "$SOURCE_NODE" ] && (  echo rsync -avn -e ssh --exclude-from=$DYW/rsync-exclusion-list.txt $DYW/ $X:$DYW_X/   ) || echo "cannot dyw-sync on node $LOCAL_NODE " 
      [ "$LOCAL_NODE" == "$SOURCE_NODE" ] && (       rsync -av  -e ssh --exclude-from=$DYW/rsync-exclusion-list.txt $DYW/ $X:$DYW_X/   ) || echo "cannot dyw-sync on node $LOCAL_NODE " 
   fi	  
}


dym-x-sync(){
   X=${1:-$TARGET_TAG}
   vname="DYM_$X"
   eval DYM_X=\$$vname

   if [ "X$DYM_X" == "X" ]; then
	  echo "to syncronise to node $X must set up the $vname variable in .bash_dyw_env " 
   else
      echo "syncronise $DYM  local copy of macros   to $X:$vname ie $DYW_X  ... first a dry run.. then the real thing "
      [ "$LOCAL_NODE" == "$SOURCE_NODE" ] && (  ssh $X mkdir -p $DYM_X ) || echo "cannot create macro folder $DYM_X on node $X "
      [ "$LOCAL_NODE" == "$SOURCE_NODE" ] && (  echo rsync -avn -e ssh --exclude-from=$DYM/rsync-exclusion-list.txt $DYM/ $X:$DYM_X/   ) || echo "cannot dyw-sync on node $LOCAL_NODE " 
      [ "$LOCAL_NODE" == "$SOURCE_NODE" ] && (       rsync -av  -e ssh --exclude-from=$DYM/rsync-exclusion-list.txt $DYM/ $X:$DYM_X/   ) || echo "cannot dyw-sync on node $LOCAL_NODE " 
   fi 
}



