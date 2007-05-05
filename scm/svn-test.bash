




svn-wc(){  $SCM_HOME/svn-wc-crawl.py $*  ; }
svn-rp(){  $SCM_HOME/svn-rp-crawl.py $*  ; }
svn-rp-ra(){  $SCM_HOME/svn-rp-ra-crawl.py $*  ; }
svn-wc-x(){  scp $SCM_HOME/svn-wc-crawl.py  ${1:-$TARGET_TAG}:$SCM_BASE ; }
svn-rp-x(){  scp $SCM_HOME/svn-rp-crawl.py  ${1:-$TARGET_TAG}:$SCM_BASE ; }


svn-wc-revision(){    
    #
	# this returns the revision number of wc_path passed (defaults to PWD) if it is a clean revision ... if dirty return -1
    #  enhancemwents:
    #       exit with error status when not clean, and dump the svn status -u    output
    #       otherwise just return the revision number
#    
#  in real case ... will be many "?" unmanaged files... even after a commit
# ??? hmm is there an svnignore functionaity ???
#     svn status -u --xml
#
#
#   maybe
#       svnversion 
#
#
	#
	#
    #
	wc_path=${1:-.}
	#echo wc_path $wc_path
	#svn status -u $wc_path
	wc_rev=$(svn status -u $wc_path | perl -n -e 'm/^Status against revision:\s*(\d*)/ && $.==1 && print $1')
	ret=${wc_rev:--1}
	echo $ret
}

