#!/bin/sh

pkg=bitten
vers=trac-0.11  ## corresponds to the leaf of the checkout  

name=$(basename $0)
base=$(dirname $0)
msg="=== $name :"

dir=$base/$vers
cd $dir

echo $msg reverting $pkg $vers in $dir
svn revert .
rev=$(svnversion $dir)
path=$base/patch/$pkg-$vers-$rev.patch

if [ -f "$path" ]; then
   echo $msg applying patch $path for revision $rev 
   patch -p0 < $path
else
   echo $msg there is no patch file at $path for revision $rev 
fi

echo $msg easy_install into the python in path
easy_install .








