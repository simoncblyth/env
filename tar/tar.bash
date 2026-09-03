tar-vi(){ vi $BASH_SOURCE ; }
tar-env(){ echo -n ; }
tar-usage(){ cat << EOU
tar
====


Extracted wildcarded single file from archive to stdout
---------------------------------------------------------

::

    tar -xOf oj_releases_el9_amd64_gcc15_J26_4_1_Opticks-v0_6_9_Thu.tar --wildcards '*/envset.sh'



EOU
}


