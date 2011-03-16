# === func-gen- : svn/svnprecommit fgp svn/svnprecommit.bash fgn svnprecommit fgh svn
svnprecommit-src(){      echo svn/svnprecommit.bash ; }
svnprecommit-source(){   echo ${BASH_SOURCE:-$(env-home)/$(svnprecommit-src)} ; }
svnprecommit-vi(){       vi $(svnprecommit-source) ; }
svnprecommit-env(){      elocal- ; }
svnprecommit-usage(){
  cat << EOU
     svnprecommit-src : $(svnprecommit-src)
     svnprecommit-dir : $(svnprecommit-dir)


     svnprecommit-create
         creates a test SVN repository
     svnprecommit-checkout
         checkout the test SVN repository
     svnprecommit-populate
         checkin a few commits
     svnprecommit-placehook
         write the pre-commit hook and make executable
     svnprecommit-testhook
         attempt to make a commit .. that should be denied

  == pre-commit hook examples ==
   
     * http://svn.apache.org/repos/asf/subversion/trunk/tools/hook-scripts/
     * http://svn.apache.org/repos/asf/subversion/trunk/contrib/hook-scripts/

  == Approaches ==

    * base on svnlook commandline
    * use python-svn bindings 



EOU
}
svnprecommit-dir(){ echo $(local-base)/env/svn/svn-svnprecommit ; }
svnprecommit-cd(){  cd $(svnprecommit-dir); }
svnprecommit-mate(){ mate $(svnprecommit-dir) ; }
svnprecommit-get(){
   local dir=$(dirname $(svnprecommit-dir)) &&  mkdir -p $dir && cd $dir
}

svnprecommit-base(){  echo /tmp ; }
svnprecommit-rpath(){ echo $(svnprecommit-base)/testrepo ; }
svnprecommit-wpath(){ echo $(svnprecommit-base)/wc_testrepo ; }
svnprecommit-hpath(){ echo $(svnprecommit-rpath)/hooks/pre-commit ; }

svnprecommit-rurl(){  echo file://$(svnprecommit-rpath) ; }
svnprecommit-name(){  echo $(basename $(svnprecommit-rurl)) ; }

svnprecommit-test(){

   echo deletes and recreates repositories and working copy 
   echo and populates ... establishing test fixture

   svnprecommit-create
   svnprecommit-checkout
   svnprecommit-populate

   svnprecommit-placehook
   svnprecommit-testhook
   
}


svnprecommit-create(){
   local msg="=== $FUNCNAME :"
   local rpath=$(svnprecommit-rpath)
   [ ${#rpath} -lt 10 ] && echo $msg SANITY CHECK FAILURE && return 1
   [ -d "$rpath" ] && echo $msg deleting repository $rpath  && rm -rf $rpath
   local cmd="svnadmin create $rpath"
   echo $msg $cmd 
   eval $cmd
}

svnprecommit-checkout(){
   local wpath=$(svnprecommit-wpath)
   local wname=$(basename $wpath)
   [ ${#wpath} -lt 10 ] && echo $msg SANITY CHECK FAILURE && return 1
   [ -d "$wpath" ] && echo $msg deleting working copy $wpath  && rm -rf $wpath
   local iwd=$PWD
   cd $(dirname $wpath)
   local cmd="svn checkout $(svnprecommit-rurl) $wname "
   echo $msg $cmd 
   eval $cmd
   cd $iwd
}

svnprecommit-wcd(){
   cd $(svnprecommit-wpath)
} 

svnprecommit-populate(){
   local wpath=$(svnprecommit-wpath)
   local wname=$(basename $wpath)
   cd $(dirname $wpath)
   touch $wname/dummy.txt
   svn add $wname/dummy.txt
   svn ci $wname/ -m "first commit "  
   svn up $wname
   svn log $wname
   echo hello >> $wname/dummy.txt
   svn ci $wname -m "second commit "
}

svnprecommit-placehook(){
  local hpath=$(svnprecommit-hpath)
  svnprecommit-demodeny- > $hpath
  cat $hpath
  chmod u+x $hpath 
}

svnprecommit-demodeny-(){ cat << EOD
#!/usr/bin/env python
import sys
sys.stderr.write("WHA WHA OOPS ... YOUR COMMIT IS NOT ALLOWED")
sys.exit(1)
EOD
}

svnprecommit-testhook(){
  svnprecommit-wcd
  echo hello >> dummy.txt
  svn ci -m "$FUNCNAME"

}
