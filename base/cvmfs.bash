cvmfs-env(){ echo -n ; }
cvmfs-vi(){ vi $BASH_SOURCE ; }
cvmfs-usage(){ cat << EOU
cvmfs
=======

See hcvmfs- for publishing to cvmfs 


processes
---------

::

    [blyth@localhost cvmfs]$ ps aux | grep cvmfs
    cvmfs     15730  0.0  0.0  60932   716 ?        S    Aug26   0:00 /usr/bin/cvmfs2 __cachemgr__ . 7 8 4194304000 2097152000 0 3 -1 :
    cvmfs     15733  0.0  0.0  73328  4020 ?        S    Aug26   1:52 /usr/bin/cvmfs2 __cachemgr__ . 7 8 4194304000 2097152000 0 3 -1 :
    cvmfs     15754  0.0  0.0 725492 30644 ?        Sl   Aug26   0:19 /usr/bin/cvmfs2 -o rw,system_mount,fsname=cvmfs2,allow_other,grab_mountpoint,uid=986,gid=979 cvmfs-config.cern.ch /cvmfs/cvmfs-config.cern.ch
    cvmfs     15758  0.0  0.0  92152 24936 ?        S    Aug26   0:00 /usr/bin/cvmfs2 -o rw,system_mount,fsname=cvmfs2,allow_other,grab_mountpoint,uid=986,gid=979 cvmfs-config.cern.ch /cvmfs/cvmfs-config.cern.ch
    cvmfs     15816  4.7  0.0 4357476 49064 ?       Sl   Aug26 207:15 /usr/bin/cvmfs2 -o rw,system_mount,fsname=cvmfs2,allow_other,grab_mountpoint,uid=986,gid=979 juno.ihep.ac.cn /cvmfs/juno.ihep.ac.cn
    cvmfs     15820  0.0  0.0  92148 24936 ?        S    Aug26   0:00 /usr/bin/cvmfs2 -o rw,system_mount,fsname=cvmfs2,allow_other,grab_mountpoint,uid=986,gid=979 juno.ihep.ac.cn /cvmfs/juno.ihep.ac.cn
    blyth    341490  0.0  0.0 112824   980 pts/2    S+   09:37   0:00 grep --color=auto cvmfs


install
-------

https://cvmfs.readthedocs.io/en/stable/cpt-quickstart.html



EOU
}
