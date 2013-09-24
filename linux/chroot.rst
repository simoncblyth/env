chroot
========


chroot needs root, but maybe not
--------------------------------

* http://stackoverflow.com/questions/3737008/how-to-run-a-command-in-a-chroot-jail-not-as-root-and-without-sudo


* :google:`sudo setcap cap_sys_chroot+ep /usr/sbin/chroot`


setcap : set capability
~~~~~~~~~~~~~~~~~~~~~~~~

::

    [blyth@belle7 ~]$ rpm -ql libcap
    /lib/libcap.so.1
    /lib/libcap.so.1.10
    /usr/sbin/execcap
    /usr/sbin/getpcaps
    /usr/sbin/setpcaps
    /usr/sbin/sucap
    /usr/share/doc/libcap-1.10
    /usr/share/doc/libcap-1.10/capability.notes
    /usr/share/doc/libcap-1.10/capfaq-0.2.txt
    [blyth@belle7 ~]$ 




chroot needs a carefully prepared environment
----------------------------------------------

* http://ubuntuforums.org/showthread.php?t=1434781


::

    chroot: /bin/sh: No such file or directory



