CMS02
======

Jun 19, 2014 : httpd offline, OOM again
-----------------------------------------

#. valmon monitoring indicates apache SVN fail  
#. no httpd, pingable but cannot SSH in 
#. ~11:00 reboot 

   * restores SSH access
   * but httpd does not come back automatically ? 


Valmon monitoring 
~~~~~~~~~~~~~~~~~~~~

::

    curl -s --connect-timeout 3 http://dayabay.phys.ntu.edu.tw/repos/env/ 


Original Cause, httpd OOM
~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    [root@cms02 log]# grep oom messages
    Jun 19 06:40:41 cms02 kernel: oom-killer: gfp_mask=0x1d2
    Jun 19 07:28:57 cms02 kernel: oom-killer: gfp_mask=0xd2
    Jun 19 08:04:39 cms02 kernel: oom-killer: gfp_mask=0xd0
    Jun 19 08:38:38 cms02 kernel: oom-killer: gfp_mask=0x1d2
    Jun 19 08:40:13 cms02 kernel: oom-killer: gfp_mask=0x1d2
    Jun 19 09:21:33 cms02 kernel: oom-killer: gfp_mask=0x1d2
    Jun 19 10:21:45 cms02 kernel: oom-killer: gfp_mask=0xd2
    Jun 19 10:27:25 cms02 kernel: oom-killer: gfp_mask=0xd2
    Jun 19 10:29:30 cms02 kernel: oom-killer: gfp_mask=0x1d2
    Jun 19 10:56:40 cms02 kernel: oom-killer: gfp_mask=0x1d2
    [root@cms02 log]# 


Restart httpd
~~~~~~~~~~~~~~~~

::

    [root@cms02 log]# /sbin/service httpd start


