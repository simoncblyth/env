svn-tools-get(){

   local dir=$LOCAL_BASE/svn
   [ -d $dir ] || $SUDO mkdir -p $dir && $SUDO chown $USER $dir   
   cd $dir
   [ -d "tools" ] && svn update tools || svn co http://svn.collab.net/repos/svn/trunk/tools/

}