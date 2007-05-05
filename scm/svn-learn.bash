
#
#   svn-learn-create-repo
#   svn-learn-create
#   svn-learn-import
#
#




#
#
#checking for JDK... configure: WARNING: no JNI header files found.
#no
#checking for perl... /usr/bin/perl
#checking for ruby... none
#checking for swig... none
#configure: Configuring python swig binding
#checking for Python includes... -I/usr/include/python2.2
#
#
#
#
# configure: WARNING: we have configured without BDB filesystem support
#
# You don't seem to have Berkeley DB version 4.0.14 or newer
# installed and linked to APR-UTIL.  We have created Makefiles which
# will build without the Berkeley DB back-end; your repositories will
# use FSFS as the default back-end.  You can find the latest version of
# Berkeley DB here:
#   http://www.sleepycat.com/download/index.shtml
#


svn-learn-create-repo(){

   name=${1:-$SVN_REPO_NAME}	
   mkdir -p $SVN_PARENT_PATH
   
   # the base must exist 
   svnadmin create  $SVN_PARENT_PATH/$name
}


svn-learn-create(){
   base=/tmp/scb/svntest
   mkdir -p $base
   # the base must exist 
   svnadmin create  $base/repo
}


svn-learn-import(){

   repo=/tmp/scb/svntest/repo
   toimp=/tmp/scb/svntest/toimp
   mkdir -p $toimp/{branches,tags,trunk}
   echo "hello" > $toimp/trunk/hello 

   svn import $toimp file://$repo -m "initial import"

}



svn-learn-remote-import(){

   svn import johnny http://grid1.phys.ntu.edu.tw:6060/repos/red/trunk/  -m "initial import" --password secret --username blyth
# the content of folder johnny is put under the trunk , not the folder johnny itself

## to "replicate" the  folder johnny into the repository do this 
   svn import johnny http://grid1.phys.ntu.edu.tw:6060/repos/red/trunk/johnny   -m "initial import" 

## user/pass not needed every time as cached locally ? ~/.svn

}



svn-learn-checkout-commit(){


   cd /tmp/scm
   ## creates folder "project"
   ##svn checkout  file://$repo/trunk project

   svn checkout http://grid1.phys.ntu.edu.tw:6060/repos/red/
   ## creates folders red/{trunk,tags,branches}/

    cd red/trunk/perl/SCB/Workflow
	vi PATH.pm
	svn commit PATH.pm
    ##  succeeds to make the revision ... thats visible thru trac

   
   ## more usual to checkout the trunk like so :
   svn checkout http://grid1.phys.ntu.edu.tw:6060/repos/red/trunk/

   
}


svn-learn-copy(){

   ## succeeded to fail for a dyw-novice but succeed for dyw-user

   red=http://grid1.phys.ntu.edu.tw:6060/repos/red	
   svn copy $red/trunk/perl/SCB/Workflow/PATH.pm $red/trunk/perl/SCB/Workflow/PATHCopy.pm -m "first copy"

}

