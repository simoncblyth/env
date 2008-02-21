svn-tools-get(){

   local dir=$LOCAL_BASE/svn
   [ -d $dir ] || $SUDO mkdir -p $dir && $SUDO chown $USER $dir   
   cd $dir
   [ -d "tools" ] && svn update tools || svn co http://svn.collab.net/repos/svn/trunk/tools/

}


svn-tools-fill-tmpl(){

   cd $LOCAL_BASE/svn/tools/backup
   perl -p -e 's/\@SVN_BINDIR\@/\/usr\/bin/g' hot-backup.py.in  > hot-backup.py
   diff hot-backup.py.in hot-backup.py

   which svnlook
   which svnadmin
}