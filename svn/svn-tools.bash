svn-tools-vi(){ vi $BASH_SOURCE ; }
svn-tools-usage(){
  cat << EOU

      svn-tools-dir : $(svn-tools-dir)
      svn-tools-url : $(svn-tools-url)
      svn-tools-hotbackup : $(svn-tools-hotbackup)


      svn-tools-build
          do the below

      svn-tools-get
          checkout or update     
      svn-tools-prepare
          fill in the template to create the hotbackup tool



EOU 

}

svn-tools-env(){

   elocal-
   svn-
}


svn-tools-dir(){       echo $(local-base)/env/svn ; }
svn-tools-url(){       echo http://svn.collab.net/repos/svn/trunk/tools/ ; }
svn-tools-hotbackup(){ echo $(svn-tools-dir)/backup/hot-backup.py ; } 


svn-tools-build(){

   svn-tools-get
   svn-tools-prepare
}


svn-tools-get(){

   local dir=$(svn-tools-dir)
   [ ! -d $dir ] && $SUDO mkdir -p $dir && $SUDO chown $USER $dir   
   cd $dir
   [ -d "tools" ] && svn update tools || svn co 

}


svn-tools-prepare(){

   local msg="=== $FUNCNAME :"
   local iwd=$PWD
   local path=$(svn-tools-hotbackup)
   local dir=$(dirname $path)
   local name=$(basename $path)

   [ ! -f "$dir" ] && echo $msg ABORT no $dir && return 1

   cd $dir
   
   perl -p -e 's/\@SVN_BINDIR\@/\/usr\/bin/g' $name.in  > $name   
   echo $msg creating $name from $name.in in $dir

   diff $name.in $name
   chmod u+x $name

   which svnlook
   which svnadmin
   cd $iwd
}
