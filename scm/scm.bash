#
#
#
#  STEPS TO BRING A FOLDER CALLED "name" UNDER MANAGEMENT
# 
#     1) cd /path/to/name 
#     2) scm-create
#     3) cd .. ; mv name name-safety
#     4) svn co http://localhost/repos/name/trunk name
#     5)  Xcode will notice the change      
#
#
#
#  ISSUES: 
#      - what system user/group(s) should own  
#            $SCM_FOLD/repos  <--- root for protection, all access goes thru svn interface 
#            $SCM_FOLD/tracs  <--- www , to match apache
#
#        cd $SCM_FOLD && $SUDO chown $APACHE2_USER:$APACHE2_GROUP tracs
#
#
#   NEXT:
#          2) real dyw import 
#
#            b)
#                first revisions for localization ... hmm potentially
#                the source is used on multiple machines, with different
#                localization needs
#
#            c)
#                building 
#
#            3)  svnversion integrate into condor logging  
#
#
#
#   scm-vi
#   scm-x-pkg
#   scm-x
#   scm-i
#   scm-ls
#
#   scm-use-create-local  name [fold] [init]  
#
#   scm-checkout name
#                              check out remote repository into pwd
#                       
#
#   scm-add-user [name]        add a user to the remote repository 
#                              [name] defaults to $USER ... if the user
#                              already exists then allows the password to be
#                              changed
#
#   scm-create  [fold] [name]
#                              create a remote repository on target node
#                              with the contents of [fold] ( defaults to pwd ) 
#                              and name [name] defaults to basename of [fold] 
#
#       scm-tracname-available
#          scm-traclist-get
#          scm-parse-traclist
#       
#
#   scm-remove [name]
#                          remove remote repository , name defaults to basename of pwd
#
#   scm-import    name fold user pass [url]
#
#       NB  this is usually invoked for you by scm-create
#
#                              import contents of "fold" into to "name"
#                              repository  at "url" , the url defaults to 
#                                  http://SCM_NODE:SCM_PORT
#                              and the import goes into /repos/$name/trunk/
#    
#
#
#
#   scm-use-remove-local name
#   scm-open        [name]
#
#
#    note to create a remote repository and tracitory (on the target note) do :
#
#           scm-create-x H green
#           scm-import     green relative-path-of-folder-to-upload
#
#   
#
#      getting permissions issues with apache2 ... arising
#      from the "shortcut" copying of the apache2 tarball  
#      ... so need to start again almost ... 
#
#      building or installing as root has consequences for the port ... 80 
#
#
#
# [g4pb:~] blyth$ svn import johnny http://grid1.phys.ntu.edu.tw:6060/repos/yellow/trunk/ -m "initial import"
# svn: MKACTIVITY of '/repos/yellow/!svn/act/0b4101f1-fe2e-0410-b312-8d38c72169a9': 403 Forbidden (http://grid1.phys.ntu.edu.tw:6060)
#
#
 scm_iwd=$(pwd)




 SCM_BASE=$ENV_BASE/scm
 export SCM_HOME=$HOME/$SCM_BASE
[ "$SCM_DBG" == "1" ] && echo $SCM_BASE/scm.bash

 cd $SCM_HOME

 [ -r scm-use.bash ]        && . scm-use.bash
 [ -r scm-backup.bash ]     && . scm-backup.bash 
 [ -r svn-use.bash ]        && . svn-use.bash 
 [ -r trac/trac-use.bash ]  && . trac/trac-use.bash 
 [ -r modwsgi-use.bash ]    && . modwsgi-use.bash 

 ## caution must exit with initial directory 
 cd $scm_iwd
 [ -t 0 ] || return 
 cd $SCM_HOME

 [ -r cvs.bash  ]           && . cvs.bash 
 [ -r file.bash  ]          && . file.bash

 [ -r apache2.bash ]        && . apache2.bash
 [ -r python.bash ]         && . python.bash
 [ -r sqlite.bash ]         && . sqlite.bash

 [ -r swig.bash ]           && . swig.bash                ## depends on python 
 [ -r svn.bash ]            && . svn.bash                 ## depends on apache2, swig, python
 [ -r svn-build.bash ]      && . svn-build.bash           ## depends on apache2, swig, python
#[ -r svn-test.bash ]       && . svn-test.bash           ## depends on apache2, swig, python
 
 [ -r pysqlite.bash ]       && . pysqlite.bash            ## depends on python, sqlite
 [ -r modpython.bash ]      && . modpython.bash           ## depends on python, apache2
 [ -r modwsgi.bash ]        && . modwsgi.bash             ## even when not using, create the .wsgi files 

 [ -r pymysql.bash ]        && . pymysql.bash

 [ -r clearsilver.bash ]                 && . clearsilver.bash         
 
 [ -r trac/trac.bash ]                   && . trac/trac.bash     
#[ -r svn-learn.bash ]      && . svn-learn.bash
#[ -r modpython-test.bash ] && . modpython-test.bash  

 ## caution must exit with initial directory
 cd $scm_iwd
 
 pexpect(){  [ -r $SCM_HOME/pexpect.bash ] && . $SCM_HOME/pexpect.bash ; } 
 svn-sync(){ [ -r $SCM_HOME/svn-sync.bash ] && . $SCM_HOME/svn-sync.bash ; } 


scm-vi(){
  iwd=$(pwd)	
  cd $HOME/$SCM_BASE 
  vi *
  cd $iwd
}


scm-x-pkg(){ 
   cd $HOME 	
   tar zcvf scm.tar.gz $SCM_BASE/*
   scp scm.tar.gz ${1:-$TARGET_TAG}:; 
   ssh ${1:-$TARGET_TAG} "tar zxvf scm.tar.gz" 
}
scm-x(){ scp $SCM_HOME/scm.bash ${1:-$TARGET_TAG}:$SCM_BASE; }
scm-i(){ .   $SCM_HOME/scm.bash  ; }




scm-ls(){
     ls  -l $SCM_FOLD/{tracs,repos}
}



scm-cmd-x(){
   ## demo of running a multi argument command on a remote node	
   X=${1:-$TARGET_TAG}	
   shift
   echo ssh $X "bash -lc \"scm-cmd $*\""
        ssh $X "bash -lc \"scm-cmd $*\""
}

scm-cmd(){
   echo scm-cmd on NODE_TAG:$NODE_TAG args:$*  
}



scm-vi(){
  iwd=$(pwd)	
  cd $HOME/$SCM_BASE 
  vi *.bash
  cd $iwd
}


scm-checkout(){

  local name=${1:-dummy}
  local branch=${2:-trunk}
  
  uurl=$SCM_URL/repos/${name}/${branch}/	 

  [ -d ".svn" ] && echo error this should be used for initial checkouts only && return 1 

  ## count the number of items in the pwd
  declare -a dirs
  dirs=($(ls -1))
  [ ${#dirs[@]} -gt 0 ] && echo must checkout into an empty directory && return 1


  echo ======== checkout $uurl into empty directory $(pwd)
  svn checkout $uurl .

}


scm-add-user(){

  name=${1:-$USER}
  [ "X$SCM_TAG" == "X" ] && ( echo ERROR ...  SCM_TAG must be defined in env/base/local.bash &&  return 1 )

  if [ "$NODE_TAG" == "$SCM_TAG" ]; then

          $ASUDO bash -lc "svn-use-apache2-add-user $name "
  else

     echo ssh $SCM_TAG "bash -lc \"svn-use-apache2-add-user $name\""
          ssh $SCM_TAG "bash -lc \"svn-use-apache2-add-user $name\""

  fi

}

scm-remove(){

   #
   #   usage:
   #         scm-remove [name]
   #   
   #     remove remote repository [name]  the name defaults to the basename of
   #     the pwd
   #

  X=$SCM_TAG
  name=${1:-$(basename $(pwd))}
   
  echo ========== scm-remove on node $X name $name 

  if [ "$NODE_TAG" == "$SCM_TAG" ]; then
     echo scm-use-remove-local $name
  else	  
     echo ssh $X "bash -lc \"scm-use-remove-local $name\""
  fi 
   
   read -n 1 -p "========== enter \"Y\" to confirm : " confirm

   if [ "$confirm" == "Y" ]; then
		echo " ================  OK proceeding as confirm is [$confirm] "
        if [ "$NODE_TAG" == "$SCM_TAG" ]; then
           scm-use-remove-local $name
        else
           ssh $X "bash -lc \"scm-use-remove-local $name\""
	    fi		
   else
	    echo ================  you chickened out , as confirm is [$confirm] rather than Y
   fi

}


scm-unpack(){

   #  TO IMPLEMENT :
   #
   #    if passed the path to a package, then unpack it into a tmpdir and place the content of the package container folder
   #    into a trunk,branches,tags heirarchy and import into the scm named after the package container folder
   #    then delete the tmpdir make a working copy directory and checkout the repository  
   #

   local pkg=$1
   [ -f "$pkg" ] || return 1
    
   local X=$SCM_TAG
     
   declare -a tlds
   local tlds=($(file-package-topdir $pkg))
   local ntld=${#tlds[@]}
   [ "$ntld" != "1" ] && echo === scm-unpack ABORTING unpack of $pkg as are $ntld top level folders: ${tlds[@]} && return 1
      
   local tld=${tlds[0]}
   echo === scm-unpack proceeding to unpack $pkg with single tld $tld === 
               
   local tmpdir=/tmp/scm-unpack/$$   
   mkdir -p $tmpdir/{unpack,trunk,branches,tags}  
       
   if [ "${pkg:(-4):4}" == ".zip" ]; then
      unzip -d $tmpdir/unpack $pkg
   elif ([ "${pkg:(-7):7}" == ".tar.gz" ] || [ "${pkg:(-4):4}" == ".tgz" ]) then
      tar -C $tmpdir/unpack -zxvf $pkg
   else
      echo unpacking $pkg is not supported && return 1
   fi
       
   cp -Rp $tmpdir/unpack/$tld/* $tmpdir/trunk/
   rm -rf $tmpdir/unpack
      

   local name=$tld 
   scm-tracname-available $name && echo proceeding as tracname $name is available || ( echo a repository named $name exists already , cannot overwrite  && return 1  )
   
   
   #local uurl=$SCM_URL
   #svn import $tmpdir $uurl/repos/$name/trunk/  -m "initial scm-unpack " --username $SCM_USER --password $NON_SECURE_PASS
   #svn checkout $uurl/repos/$name/trunk/ $name
}



scm-create(){

   #
   #   usage:
   #         scm-create [fold] [name]
   #   
   #     
   #     create repository and tracitory on the SCM_TAG remote node 
   #     and upload the contents of the local folder [fold] into its
   #     trunk
   # 
   # 
   #     [fold] defaults to the pwd
   #     [name] defaults to the basename of fold 
   #
     #

   X=$SCM_TAG

   fold=${1:-$(pwd)}
   name=${2:-$(basename $fold)}

   #
   # access the list of repositories , to ensure not stamping on a preexisting one
   #
   scm-tracname-available $name && echo proceeding as tracname $name is available || ( echo a repository named $name exists already , cannot overwrite  && return 1  )
   #
   # can access $? rather than doing it in a pipe

  
   echo ========== scm-create targetting node $X fold $fold name $name 

   if [ "$SCM_TAG" == "$NODE_TAG" ]; then
     echo scm-use-create-local $name
   else	 
     echo ssh $X "bash -lc \"scm-use-create-local $name\""
   fi
   
   read -n 1 -p "========== enter \"Y\" to confirm : " confirm

   if [ "$confirm" == "Y" ]; then
	    
		echo " ================  OK proceeding as confirm is [$confirm] "
       
        if [ "$SCM_TAG" == "$NODE_TAG" ]; then
           scm-use-create-local $name
		else	
	       ssh $X "bash -lc \"scm-use-create-local $name\""
        fi


         
        ##  this import goes thru apache ...
        if [ "X$fold" == "XEMPTY" ]; then
            echo skip the import as fold is $fold
        else
            scm-import $name $fold
		fi
        
        echo ============== check the repository by visiting $SCM_URL/tracs/$name/browser/trunk/
        #scm-open   $name


   else
	    echo ================  you chickened out , as confirm is [$confirm] rather than Y
   fi
		
}


scm-tracname-available(){
  
  want=${1:-dummy}
  traclist=/tmp/traclist-$$.html
  
  scm-traclist-get
  
  [ -f "$traclist" ] || ( echo there is no traclist $traclist ...  scm-traclist-get must be preceeded in each process by running scm-traclist-get && return 1 )
 
  tracs=$(scm-parse-traclist $traclist) 

  for t in $tracs
  do
	 [ "$t" == "$want" ] && echo "found match trac [$t] for argument [$want] " && return 1    
  done	 
  echo no match found for argument [$want]
  return 0
  
  ##  see the result in $? after invokation  ... not finding a match is success
}

scm-parse-traclist(){ perl -n -e 'm|href=\"/tracs/(\S*)\"| && print "$1 "' ${1:-dummy} ; }

scm-traclist-get(){


 traclist=/tmp/traclist-$$.html
 lurl=$SCM_URL/tracs/

 echo ======= access $lurl to discover the repositories already present 
 curl -o $traclist $lurl 
 echo ======= examine the html $traclist to determine the repositories
 cat $traclist
 tracs=$(scm-parse-traclist $traclist) 

 echo ====== list the repositories
 for trac in $tracs
 do
	 echo $trac    
 done	 

}


scm-import(){

  turl="$SCM_URL"
  
  name=${1:-dummy}
  fold=${2:-dummy}
  user=${3:-$SCM_USER}
  pass=${4:-$SCM_PASS}
  uurl=${5:-$turl}

  echo ======= scm/trac/scm.bash::scm-import starting ========= 

  
  [ $name == "dummy" ] && ( echo argument1 should be a remote repository name at $uurl && return )
  [ -d "$fold" ]       || ( echo argument2 $fold should be a valid directory on local node && return )
 
  iwd=$(pwd)
  cd $(dirname $fold)
  echo ======= contents of folder $fold is put under the trunk at $uurl , not the folder $fold itself === will prompt for password 
  echo ======  svn import $(basename $fold) $uurl/repos/$name/trunk/  -m "initial scm-import " --username $user --password $NON_SECURE_PASS
               svn import $(basename $fold) $uurl/repos/$name/trunk/  -m "initial scm-import " --username $user --password $NON_SECURE_PASS

  cd $iwd			   


   echo ======= scm/trac/scm.bash::scm-import finished ========= 

#
#[g4pb:~] blyth$  svn import env http://hfag.phys.ntu.edu.tw:6060/repos/green/trunk/ -m 'initial scm-import ' 
#Authentication realm: <http://hfag.phys.ntu.edu.tw:6060> svn-repos
#Password for 'blyth': 
#svn: MKACTIVITY of
#'/repos/green/!svn/act/5c0fc465-7f2f-0410-9768-97e36de2abff': 403 Forbidden
#(http://hfag.phys.ntu.edu.tw:6060)
#


}






scm-open(){

   name=${1:-dummy}

   ## open $SCM_URL/repos/$name/      # no-frills view of the "HEAD" being provided by the raw svn 
   ## open $SCM_URL/repos/            # discovery is not allowed

   prefix=""
   if [ "$SOURCE_TAG" != "$NODE_TAG" ]; then
      prefix="ssh $SOURCE_TAG"
   fi

   echo prefix : [$prefix]

   if [ "$name" == "dummy" ]; then
       $prefix open $SCM_URL/tracs/   
   else 	   
       $prefix open $SCM_URL/tracs/$name/browser/trunk/         # frilly view provided by trac
   fi	  

}


scm-create-with(){

  path=${1:-dummy}

  name=$(basename $path)
  fold=$(dirname $path)
  base=/tmp

  svnadmin create $base/$name
  cd $fold && svn import $name file://$base/$name
  
}


scm-import-from-cvs(){

  echo scm-import-from-cvs



  
#
#  svnadmin create /tmp/dyw
#  cd $DYW/..
#  svn import dyw file:///tmp/dyw 
#
#  svnadmin create /tmp/dyw2
#  cd $(dirname $svn import $DYW file:///tmp/dyw2 
#    absolute imports are generally not a good idea
#
#  svnadmin create /tmp/dyw3
#
#
#  mkdir /tmp/co
#  cd /tmp/co
#  svn checkout file:///tmp/dyw
#
#   creates /tmp/co/dyw
#


}


