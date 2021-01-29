# === func-gen- : tools/centos fgp tools/centos.bash fgn centos fgh tools src base/func.bash
centos-source(){   echo ${BASH_SOURCE} ; }
centos-edir(){ echo $(dirname $(centos-source)) ; }
centos-ecd(){  cd $(centos-edir); }
centos-dir(){  echo $LOCAL_BASE/env/tools/centos ; }
centos-cd(){   cd $(centos-dir); }
centos-vi(){   vi $(centos-source) ; }
centos-env(){  elocal- ; }
centos-usage(){ cat << EOU


CentOS 
=========

* https://danielmiessler.com/study/fedora_redhat_centos/


Fedora is the main project, and it’s a communitity-based, free distro
focused on quick releases of new features and functionality.

Redhat is the corporate version based on the progress of that project, and
it has slower releases, comes with support, and isn’t free.

CentOS is basically the community version of Redhat. So it’s pretty much
identical, but it is free and support comes from the community as opposed to
Redhat itself.

* https://en.wikipedia.org/wiki/CentOS

Relation between RHEL version and CentOS is one-to-one



=========  ======= =====  ============  ==================  =====================  ============
CentOS      arch   RHEL    Kernel        Centos release       RHEL release          days delay
=========  ======= =====  ============  ==================  =====================  ============
7.0-1406 	x86-64  7.0 	3.10.0-123 	7 July 2014 	     10 June 2014 	        27
7.1-1503 	x86-64 	7.1 	3.10.0-229 	31 March 2015  	     5 March 2015 	        26
7.2-1511 	x86-64 	7.2 	3.10.0-327 	14 December 2015 	 19 November 2015    	25
7.3-1611 	x86-64 	7.3 	3.10.0-514 	12 December 2016 	 3 November 2016    	39
7.4-1708 	x86-64 	7.4 	3.10.0-693 	13 September 2017 	 31 July 2017           43
7.5-1804 	x86-64 	7.5 	3.10.0-862 	10 May 2018     	 10 April 2018          31
7.6-1810 	x86-64 	7.6 	3.10.0-957 	3 December 2018 	 30 October 2018        34 
=========  ======= =====  ============  ==================  =====================  ============


change background color of existing terminal from commandline
------------------------------------------------------------------

* man gconftool-2
* https://bugzilla.gnome.org/show_bug.cgi?id=569869


screenshots
--------------

* https://www.wikihow.com/Take-a-Screenshot-in-Linux


Of the active window
~~~~~~~~~~~~~~~~~~~~~~~

1. press Fn+PrintScreen   (Fn: blue key to bottom left)


Of a selection
~~~~~~~~~~~~~~~~~~~~

1. press : Shift+Fn+PrintScreen  makes cursor change to crosshairs, press escape to cancel
2. drag out a rectangle, on releasing mouse button should hear camera shutter sound 
3. a dated screenshot png should have been saved into in ~/Pictures



enabling fullscreen of any window in GNOME 
------------------------------------------------

* Settings > Devices > Keyboard > [Keyboard Shortcuts] 

  * near the bottom of the list click "Toggle fullscreen mode" and pick a shortcut key
  * I used F11, as that matches gedit 



/etc/centos-release inconsistent with uname -a
-----------------------------------------------

* https://access.redhat.com/discussions/3160201

Jamie Bainbridge

The /etc/redhat-release file is not owned by the kernel package, it is owned by the redhat-release-server package:

So you can update redhat-release-server if you want /etc/redhat-release to say
"6.9" but you'll still be running the rest of the packages from 6.7 and have
some mix of packages from the two minor releases. We do support this but we
don't explicitly test for problems with it. If you run into an issue in future,
we may need you to update another package to resolve that issue.

It is probably better if you update all packages with yum update, though make
sure you test this in your test environment first.


verbose boot
--------------

* https://unix.stackexchange.com/questions/167521/how-do-i-make-my-boot-log-more-verbose


Tried this, seems no difference::

   [blyth@localhost ~]$ sudo vi /etc/sysconfig/init 
   [sudo] password for blyth: 


console access
---------------

ctrl-alt-f...


centos7 gcc7
-------------


::

    blyth@localhost ~]$ sudo yum search devtoolset-7-gcc
    ...
    devtoolset-7-gcc.x86_64 : GCC version 7
    devtoolset-7-gcc-c++.x86_64 : C++ support for GCC version 7
    devtoolset-7-gcc-gdb-plugin.x86_64 : GCC 7 plugin for GDB
    devtoolset-7-gcc-gfortran.x86_64 : Fortran support for GCC 7
    devtoolset-7-gcc-plugin-devel.x86_64 : Support for compiling GCC plugins

    [blyth@localhost ~]$ sudo yum install devtoolset-7-gcc devtoolset-7-gcc-c++

::

    [simon@localhost ~]$ gcc --version
    gcc (GCC) 7.3.1 20180303 (Red Hat 7.3.1-5)
    Copyright (C) 2017 Free Software Foundation, Inc.
    This is free software; see the source for copying conditions.  There is NO
    warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.





centos7 gcc8 ?
-----------------

* https://unix.stackexchange.com/questions/477360/centos-7-gcc-8-installation

Now devtools-8 is available and it's possible to use it by following commands:

::

    yum install centos-release-scl
    yum install devtoolset-8-gcc devtoolset-8-gcc-c++
    scl enable devtoolset-8 -- bash

* http://wiki.centos.org/SpecialInterestGroup/SCLo
* https://www.softwarecollections.org/en/
* https://www.softwarecollections.org/en/scls/
* https://www.softwarecollections.org/en/scls/rhscl/devtoolset-8/
* https://unix.stackexchange.com/questions/175851/how-to-permanently-enable-scl-centos-6-4

* https://access.redhat.com/documentation/en-us/red_hat_developer_toolset/8/html/8.0_release_notes/dts8.0_release

::

    [blyth@localhost ~]$ rpm -ql devtoolset-8-gcc-8.3.1-3.2.el7.x86_64
    /opt/rh/devtoolset-8/root/usr/bin/cc
    /opt/rh/devtoolset-8/root/usr/bin/cpp
    /opt/rh/devtoolset-8/root/usr/bin/gcc
    /opt/rh/devtoolset-8/root/usr/bin/gcc-ar
    /opt/rh/devtoolset-8/root/usr/bin/gcc-nm
    /opt/rh/devtoolset-8/root/usr/bin/gcc-ranlib
    /opt/rh/devtoolset-8/root/usr/bin/gcov
    /opt/rh/devtoolset-8/root/usr/bin/gcov-dump
    /opt/rh/devtoolset-8/root/usr/bin/gcov-tool
    /opt/rh/devtoolset-8/root/usr/bin/i686-redhat-linux-gcc-8
    /opt/rh/devtoolset-8/root/usr/bin/x86_64-redhat-linux-gcc

::

    [blyth@localhost ~]$ /opt/rh/devtoolset-8/root/usr/bin/gcc --version
    gcc (GCC) 8.3.1 20190311 (Red Hat 8.3.1-3)
    Copyright (C) 2018 Free Software Foundation, Inc.
    This is free software; see the source for copying conditions.  There is NO
    warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.


* https://stackoverflow.com/questions/53310625/how-to-install-gcc8-using-devtoolset-8-gcc

::

   source /opt/rh/devtoolset-8/enable 


::

    [blyth@localhost ~]$ gcc --version
    gcc (GCC) 4.8.5 20150623 (Red Hat 4.8.5-39)
    Copyright (C) 2015 Free Software Foundation, Inc.
    This is free software; see the source for copying conditions.  There is NO
    warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

    [blyth@localhost ~]$ source /opt/rh/devtoolset-8/enable 
    [blyth@localhost ~]$ which gcc
    /opt/rh/devtoolset-8/root/usr/bin/gcc
    [blyth@localhost ~]$ gcc --version
    gcc (GCC) 8.3.1 20190311 (Red Hat 8.3.1-3)
    Copyright (C) 2018 Free Software Foundation, Inc.
    This is free software; see the source for copying conditions.  There is NO
    warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.


* https://developers.redhat.com/products/developertoolset/hello-world#fndtn-macos


devtoolset-9
-------------------

::

    blyth@localhost source]$ scl enable devtoolset-9 bash
    .bashrc OPTICKS_MODE dev TERM_ORIG xterm-256color TERM xterm-256color
    /home/blyth/junotop/ExternalLibs/Opticks/0.1.0/bashrc : no OPTICKS_TOP : OPTICKS_MODE dev
    [blyth@localhost source]$ gcc --version
    gcc (GCC) 9.3.1 20200408 (Red Hat 9.3.1-2)
    Copyright (C) 2019 Free Software Foundation, Inc.
    This is free software; see the source for copying conditions.  There is NO
    warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

    [blyth@localhost source]$ 








EOU
}
centos-get(){
   local dir=$(dirname $(centos-dir)) &&  mkdir -p $dir && cd $dir

}
