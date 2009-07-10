svntools-vi(){ vi $BASH_SOURCE ; }
svntools-usage(){
  cat << EOU

      svntools-dir : $(svntools-dir)
      svntools-url : $(svntools-url)
      svntools-hotbackup : $(svntools-hotbackup)

      svntools-build
          do the below

      svntools-get
          checkout or update     
      svntools-prepare
          fill in the template to create the hotbackup tool

EOU

}

svntools-env(){
   elocal-
   svn-
}

svntools-dir(){       echo $(local-base)/env/svn ; }
svntools-url(){       echo http://svn.collab.net/repos/svn/trunk/tools/ ; }
svntools-hotbackup(){ echo $(svntools-dir)/tools/backup/hot-backup.py ; } 

svntools-build(){

   svntools-get
   svntools-prepare
}


svntools-get(){
   local dir=$(svntools-dir)
   [ ! -d $dir ] && type $FUNCNAME && $SUDO mkdir -p $dir && $SUDO chown $USER $dir   
   cd $dir

   local rev
   case "$(python -V 2>&1)" in
     "Python 2.3.4") rev=34959 ;;
                  *) rev=HEAD ;;
    esac
 
   [ -d "tools" ] && svn update -r $rev tools || svn co $(svntools-url)@$rev
}


svntools-prepare(){

   local msg="=== $FUNCNAME :"
   local iwd=$PWD
   local path=$(svntools-hotbackup)
   local dir=$(dirname $path)
   local name=$(basename $path)

   [ ! -d "$dir" ] && echo $msg ABORT no $dir && return 1
   cd $dir
   
   echo $msg creating $name from $name.in in $dir
   local svnlook=$(which svnlook)
   local svnbin=$(dirname $svnlook)

   perl -p -e "s,\@SVN_BINDIR\@,$svnbin,g" $name.in  > $name   

   diff $name.in $name
   chmod u+x $name

   echo $msg the svnbin $svnbin should match ...
   which svnlook
   which svnadmin
   cd $iwd
}
