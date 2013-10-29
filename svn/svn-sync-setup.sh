#!/bin/bash -l

usage() {

cat << EOU
  following:
    http://www.cardinalpath.com/how-to-use-svnsync-to-create-a-mirror-backup-of-your-subversion-repository/ 
  # edit the pre-revprop-change.::

    vi /home/scm/svn/dybaux/hooks/pre-revprop-change.tmpl 
    vi /home/scm/svn/dybaux/hooks/pre-revprop-change
    chmod +x /home/scm/svn/dybaux/hooks/pre-revprop-change

  # if the repo is empty::  
  svnsync initialize http://202.122.39.101/svn/dybaux http://dayabay.ihep.ac.cn/svn/dybaux --source-username lint

  # if the target is not empty::
    svn info http://dayabay.ihep.ac.cn/svn/dybaux
    svn propset --revprop -r0 svn:sync-from-uuid e4312ef9-e36e-0410-b6ce-a2946f6b7755 http://202.122.39.101/svn/dybaux
    svn propset --revprop -r0 svn:sync-last-merged-rev 5467 http://202.122.39.101/svn/dybaux
    svn propset --revprop -r0 svn:sync-from-url http://dayabay.ihep.ac.cn/svn/dybaux http://202.122.39.101/svn/dybaux
    svnsync synchronize http://202.122.39.101/svn/dybaux

EOU
}

svn-sync-initialize() {
  reponame=${1:-dybaux}
  srcserver=${2:-http://dayabay.ihep.ac.cn/svn/}
  dstserver=${3:-http://202.122.39.101/svn/}
  $FUNCNAME- $srcserver/$reponame $dstserver/$reponame
}
svn-sync-initialize-() {
  local src="$1"
  local dst="$2"

  local rev=$(svn-lastrev- $src)
  local uuid=$(svn-uuid- $src)
  
  cat << EOC
  svn propset --revprop -r0 svn:sync-from-uuid $uuid $dst
  svn propset --revprop -r0 svn:sync-last-merged-rev $rev $dst
  svn propset --revprop -r0 svn:sync-from-url $src $dst
EOC
}
