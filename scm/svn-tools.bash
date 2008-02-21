svn-tools-get(){

   local dir=$LOCAL_BASE/svn
   [ -d $dir ] || $SUDO mkdir -p $dir && $SUDO chown $USER $dir   
   cd $dir
   [ -d "tools" ] && svn update tools || svn co http://svn.collab.net/repos/svn/trunk/tools/

}


svn-tools-fill-tmpl(){

   cd $LOCAL_BASE/svn/tools/backup
   
   local name=hot-backup.py
   perl -p -e 's/\@SVN_BINDIR\@/\/usr\/bin/g' $name.in  > $name
   diff $name.in $name
   chmod u+x $name

   which svnlook
   which svnadmin
}