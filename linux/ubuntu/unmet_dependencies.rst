apt-get Unmet Dependencies
===========================

* http://askubuntu.com/questions/140246/how-do-i-resolve-unmet-dependencies
* http://askubuntu.com/questions/101802/packages-having-unmet-dependencies-broken-packages

* http://ubuntuforums.org/showthread.php?t=1476775

pinning and holding pkgs
-------------------------

* https://help.ubuntu.com/community/PinningHowto


unmets prevent apt-get installation
----------------------------------------

Cannot install anything due to unmet dependencies.

sudo apt-get install python-numpy -y
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    aracity@aracity-desktop:~$ sudo apt-get install python-numpy -y
    [sudo] password for aracity: 
    Reading package lists... Done
    Building dependency tree           
    Reading state information... Done
    You might want to run `apt-get -f install' to correct these:
    The following packages have unmet dependencies:
      libc6-dev: Depends: libc6 (= 2.9-4ubuntu6.3) but 2.12.1-0ubuntu10.2 is to be installed
      libgcc1: Depends: gcc-4.5-base (= 4.5.1-7ubuntu2) but it is not installable
    E: Unmet dependencies. Try 'apt-get -f install' with no packages (or specify a solution).
    aracity@aracity-desktop:~$ 



* :google:`libc6-dev: Depends: libc6`
* :google:`libgcc1: Depends: gcc-4.5-base`


sudo apt-get install apt-rdepends
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    aracity@aracity-desktop:~$ sudo apt-get install apt-rdepends
    Reading package lists... Done
    Building dependency tree       
    Reading state information... Done
    You might want to run `apt-get -f install' to correct these:
    The following packages have unmet dependencies:
      apt-rdepends: Depends: libapt-pkg-perl (>= 0.1.11) but it is not going to be installed
      libc6-dev: Depends: libc6 (= 2.9-4ubuntu6.3) but 2.12.1-0ubuntu10.2 is to be installed
      libgcc1: Depends: gcc-4.5-base (= 4.5.1-7ubuntu2) but it is not installable
    E: Unmet dependencies. Try 'apt-get -f install' with no packages (or specify a solution).
    aracity@aracity-desktop:~$ 


sudo apt-get -f install
------------------------

Command suggests to remove most packages on the machine. I chickened out of that.::

    aracity@aracity-desktop:~$ sudo apt-get -f install
    Reading package lists... Done
    Building dependency tree       
    Reading state information... Done
    Correcting dependencies... Done
    The following packages were automatically installed and are no longer required:
      vim-gui-common libpthread-stubs0 yorick-data cfortran libpthread-stubs0-dev
    Use 'apt-get autoremove' to remove them.
    The following packages will be REMOVED:
      acl acpi-support acpid adduser alacarte alsa-base alsa-utils anacron apparmor apparmor-utils apport apport-gtk apt apt-transport-https apt-utils apt-xapian-index aptitude apturl aspell aspell-en at at-spi avahi-autoipd avahi-daemon avahi-utils base-files base-passwd
      ... seemingly most pks on the machine...
      xserver-xorg-video-tseng xserver-xorg-video-v4l xserver-xorg-video-vesa xserver-xorg-video-vmware xserver-xorg-video-voodoo xsltproc xterm xtrans-dev xulrunner-1.9 xulrunner-1.9-gnome-support yelp yorick yorick-yeti-fftw zenity zip zlib1g zlib1g-dev
    WARNING: The following essential packages will be removed.
    This should NOT be done unless you know exactly what you are doing!
      apt libc6 (due to apt) libgcc1 (due to apt) libstdc++6 (due to apt) base-files base-passwd (due to base-files) libpam-modules (due to base-files) bash debianutils (due to bash) libncurses5 (due to bash) bsdutils coreutils libacl1 (due to coreutils) libselinux1 (due to
      coreutils) dash diff dpkg lzma (due to dpkg) e2fsprogs e2fslibs (due to e2fsprogs) libblkid1 (due to e2fsprogs) libcomerr2 (due to e2fsprogs) libss2 (due to e2fsprogs) libuuid1 (due to e2fsprogs) findutils grep libpcre3 (due to grep) gzip hostname login libpam0g (due
      to login) libpam-runtime (due to login) mktemp mount libsepol1 (due to mount) libvolume-id1 (due to mount) ncurses-base ncurses-bin perl-base python-minimal python2.6-minimal (due to python-minimal) sed sysvinit-utils tar util-linux lsb-base (due to util-linux) tzdata
      (due to util-linux) libslang2 (due to util-linux) zlib1g (due to util-linux)
    0 upgraded, 0 newly installed, 1212 to remove and 1 not upgraded.
    3 not fully installed or removed.
    After this operation, 2809MB disk space will be freed.
    You are about to do something potentially harmful.
    To continue type in the phrase 'Yes, do as I say!'
     ?] 
    Abort.


Make sense of dependencies
---------------------------

::

    aracity@aracity-desktop:~$ apt-cache showpkg python-numpy 
    Package: python-numpy
    Versions: 
    1:1.2.1-1ubuntu1 (/var/lib/apt/lists/tw.archive.ubuntu.com_ubuntu_dists_jaunty_main_binary-amd64_Packages)
     Description Language: 
                     File: /var/lib/apt/lists/tw.archive.ubuntu.com_ubuntu_dists_jaunty_main_binary-amd64_Packages
                      MD5: 7f7fe2d2ed613d0e3a7ffabfd3df14fb

    Reverse Depends: 
      python-pygpu,python-numpy
      ...
      python-gtk2,python-numpy
      inkscape,python-numpy
    Dependencies: 
    1:1.2.1-1ubuntu1 - python (3 2.7) 
                       python (2 2.5) 
                       python-central (2 0.6.11) 
                       libblas3gf (16 (null)) 
                       libblas.so.3gf (16 (null)) 
                       libatlas3gf-base (0 (null)) 
                       libc6 (2 2.4) 
                       libgcc1 (2 1:4.1.1) 
                       libgfortran3 (2 4.3) 
                       liblapack3gf (16 (null)) 
                       liblapack.so.3gf (16 (null)) 
                       libatlas3gf-base (0 (null)) 
                       python-numpy-doc (0 (null)) 
                       python-numpy-dbg (0 (null)) 
                       python-nose (2 0.10.1) 
                       python-f2py (1 2.45.241+1926-5) 
                       python-matplotlib (3 0.90.1-3) 
                       python-numpy-dev (1 1:1.0.3-2) 
                       python-scipy (1 0.6.0-6) 
                       python2.3-f2py (0 (null)) 
                       python2.4-f2py (0 (null)) 
    Provides: 
    1:1.2.1-1ubuntu1 - python2.6-numpy python2.5-numpy python-numpy-ext python-numpy-dev python-f2py 
    Reverse Provides: 



Maybe::

    sudo apt-get update





libgcc1
~~~~~~~~~~

* http://askubuntu.com/questions/154044/installing-particular-versions-when-repo-has-newer-versions

::

    aracity@aracity-desktop:~$ sudo apt-get install libgcc1
    Reading package lists... Done
    Building dependency tree       
    Reading state information... Done
    libgcc1 is already the newest version.
    You might want to run `apt-get -f install' to correct these:
    The following packages have unmet dependencies:
      libc6-dev: Depends: libc6 (= 2.9-4ubuntu6.3) but 2.12.1-0ubuntu10.2 is to be installed
      libgcc1: Depends: gcc-4.5-base (= 4.5.1-7ubuntu2) but it is not installable
    E: Unmet dependencies. Try 'apt-get -f install' with no packages (or specify a solution).


Hmm the wrong libgcc1 has been installed.::

    aracity@aracity-desktop:~$ dpkg -L libgcc1
    /.
    /lib
    /lib/libgcc_s.so.1
    /usr
    /usr/share
    /usr/share/doc
    /usr/share/lintian
    /usr/share/lintian/overrides
    /usr/share/lintian/overrides/libgcc1
    /usr/share/doc/libgcc1

Simulate removal to see how much it would break. Almost all packages::

    aracity@aracity-desktop:~$ sudo apt-get -s remove libgcc1 
    Reading package lists... Done
    Building dependency tree       
    Reading state information... Done
    You might want to run `apt-get -f install' to correct these:
    The following packages have unmet dependencies:
      apt: Depends: libgcc1 (>= 1:4.1.1) but it is not going to be installed
      apt-transport-https: Depends: libgcc1 (>= 1:4.1.1) but it is not going to be installed
      apt-utils: Depends: libgcc1 (>= 1:4.1.1) but it is not going to be installed
      aptitude: Depends: libgcc1 (>= 1:4.1.1) but it is not going to be installed
      aspell: Depends: libgcc1 (>= 1:4.1.1) but it is not going to be installed
      cmake: Depends: libgcc1 (>= 1:4.1.1) but it is not going to be installed
      cups: Depends: libgcc1 (>= 1:4.1.1) but it is not going to be installed
      cupsddk: Depends: libgcc1 (>= 1:4.1.1) but it is not going to be installed
      cupsddk-drivers: Depends: libgcc1 (>= 1:4.1.1) but it is not going to be installed
      dvd+rw-tools: Depends: libgcc1 (>= 1:4.1.1) but it is not going to be installed
      ekiga: Depends: libgcc1 (>= 1:4.1.1) but it is not going to be installed
      espeak: Depends: libgcc1 (>= 1:4.1.1) but it is not going to be installed
      evince: Depends: libgcc1 (>= 1:4.1.1) but it is not going to be installed
      firefox: Depends: libgcc1 (>= 1:4.1.1) but it is not going to be installed
      firefox-gnome-support: Depends: libgcc1 (>= 1:4.1.1) but it is not going to be installed
      gcc-3.4: Depends: libgcc1 (>= 1:3.4.6-6ubuntu3) but it is not going to be installed
      gcc-4.3: Depends: libgcc1 (>= 1:4.3.3-5ubuntu4) but it is not going to be installed
      gettext-base: Depends: libgcc1 (>= 1:4.1.1) but it is not going to be installed
      gnome-games: Depends: libgcc1 (>= 1:4.1.1) but it is not going to be installed
      gnome-system-monitor: Depends: libgcc1 (>= 1:4.1.1) but it is not going to be installed
      gnuplot-x11: Depends: libgcc1 (>= 1:4.1.1) but it is not going to be installed
      groff: Depends: libgcc1 (>= 1:4.1.1) but it is not going to be installed
      groff-base: Depends: libgcc1 (>= 1:4.1.1) but it is not going to be installed
      gstreamer0.10-plugins-good: Depends: libgcc1 (>= 1:4.1.1) but it is not going to be installed
      hal: Depends: libgcc1 (>= 1:4.1.1) but it is not going to be installed
      hpijs: Depends: libgcc1 (>= 1:4.1.1) but it is not going to be installed
      lftp: Depends: libgcc1 (>= 1:4.1.1) but it is not going to be installed
      libaspell15: Depends: libgcc1 (>= 1:4.1.1) but it is not going to be installed
      libatlas3gf-base: Depends: libgcc1 (>= 1:4.1.1) but it is not going to be installed
      libblas3gf: Depends: libgcc1 (>= 1:4.1.1) but it is not going to be installed
      ...







sudo apt-cache showpkg libgcc1
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Despite the two versions, the newer is a dead end.::

    aracity@aracity-desktop:~$ sudo apt-cache showpkg libgcc1     
    Package: libgcc1
    Versions: 
    1:4.5.1-7ubuntu2 (/var/lib/dpkg/status)
     Description Language: 
                     File: /var/lib/dpkg/status
                      MD5: bbd60d723e97d8e06c04228ee4c76f10

    1:4.3.3-5ubuntu4 (/var/lib/apt/lists/old-releases.ubuntu.com_ubuntu_dists_jaunty_main_binary-amd64_Packages)
     Description Language: 
                     File: /var/lib/apt/lists/old-releases.ubuntu.com_ubuntu_dists_jaunty_main_binary-amd64_Packages
                      MD5: bbd60d723e97d8e06c04228ee4c76f10


    Reverse Depends: 
      libc6,libgcc1
      gcc-3.4,libgcc1 1:3.4.6-6ubuntu3
      mjpegtools,libgcc1 1:4.1.1
      libmjpegtools-1.9,libgcc1 1:4.1.1
      xulrunner-1.9.2-testsuite,libgcc1 1:4.1.1
      xulrunner-1.9.1-testsuite,libgcc1 1:4.1.1
      xulrunner-1.9.1-gnome-support,libgcc1 1:4.1.1
      xulrunner-1.9.1,libgcc1 1:4.1.1
      xpdf-utils,libgcc1 1:4.1.1
      ...
    Dependencies: 
    1:4.5.1-7ubuntu2 - gcc-4.5-base (5 4.5.1-7ubuntu2) libc6 (2 2.2.5) 
    1:4.3.3-5ubuntu4 - gcc-4.3-base (5 4.3.3-5ubuntu4) libc6 (2 2.2.5) 
    Provides: 
    1:4.5.1-7ubuntu2 - 
    1:4.3.3-5ubuntu4 - 
    Reverse Provides: 








gcc-4.5-base not available 
-----------------------------

::

    aracity@aracity-desktop:~$ sudo apt-get install gcc-4.5-base
    Reading package lists... Done
    Building dependency tree       
    Reading state information... Done
    Package gcc-4.5-base is not available, but is referred to by another package.
    This may mean that the package is missing, has been obsoleted, or
    is only available from another source
    E: Package gcc-4.5-base has no installation candidate
    aracity@aracity-desktop:~$ 


Perhaps a partial attempt to install gcc 4.5 has left the tree in a funny state.

::

    aracity@aracity-desktop:~$ gcc -v  
    Using built-in specs.
    Target: x86_64-linux-gnu
    Configured with: ../src/configure -v 
        --with-pkgversion='Ubuntu 4.3.3-5ubuntu4' 
        --with-bugurl=file:///usr/share/doc/gcc-4.3/README.Bugs 
        --enable-languages=c,c++,fortran,objc,obj-c++ 
        --prefix=/usr 
        --enable-shared 
        --with-system-zlib 
        --libexecdir=/usr/lib
        --without-included-gettext
        --enable-threads=posix
        --enable-nls
        --with-gxx-include-dir=/usr/include/c++/4.3
        --program-suffix=-4.3
        --enable-clocale=gnu
        --enable-libstdcxx-debug
        --enable-objc-gc
        --enable-mpfr
        --with-tune=generic
        --enable-checking=release
        --build=x86_64-linux-gnu
        --host=x86_64-linux-gnu
        --target=x86_64-linux-gnu

    Thread model: posix
    gcc version 4.3.3 (Ubuntu 4.3.3-5ubuntu4) 
    aracity@aracity-desktop:~$ 


::

    aracity@aracity-desktop:~$ lsb_release -a
    No LSB modules are available.
    Distributor ID: Ubuntu
    Description:    Ubuntu 9.04
    Release:        9.04
    Codename:       jaunty




Downgrade to reduce unmet
---------------------------





Simulated libgcc1 downgrade reduces unmet dependencies::

    aracity@aracity-desktop:~$ sudo apt-get -s install libgcc1=1:4.3.3-5ubuntu4
    Reading package lists... Done
    Building dependency tree       
    Reading state information... Done
    You might want to run `apt-get -f install' to correct these:
    The following packages have unmet dependencies:
      libc6-dev: Depends: libc6 (= 2.9-4ubuntu6.3) but 2.12.1-0ubuntu10.2 is to be installed
    E: Unmet dependencies. Try 'apt-get -f install' with no packages (or specify a solution).


Simulated leaving at current still has two unmets::

    aracity@aracity-desktop:~$ sudo apt-get -s install libgcc1=1:4.5.1-7ubuntu2
    Reading package lists... Done
    Building dependency tree       
    Reading state information... Done
    libgcc1 is already the newest version.
    You might want to run `apt-get -f install' to correct these:
    The following packages have unmet dependencies:
      libc6-dev: Depends: libc6 (= 2.9-4ubuntu6.3) but 2.12.1-0ubuntu10.2 is to be installed
      libgcc1: Depends: gcc-4.5-base (= 4.5.1-7ubuntu2) but it is not installable
    E: Unmet dependencies. Try 'apt-get -f install' with no packages (or specify a solution).






proceed
~~~~~~~~


::

    aracity@aracity-desktop:~$ sudo apt-get  install libgcc1=1:4.3.3-5ubuntu4
    Reading package lists... Done
    Building dependency tree       
    Reading state information... Done
    You might want to run `apt-get -f install' to correct these:
    The following packages have unmet dependencies:
      libc6-dev: Depends: libc6 (= 2.9-4ubuntu6.3) but 2.12.1-0ubuntu10.2 is to be installed
    E: Unmet dependencies. Try 'apt-get -f install' with no packages (or specify a solution).



::

    aracity@aracity-desktop:~$ sudo apt-cache showpkg libc6-dev              
    Package: libc6-dev
    Versions: 
    2.9-4ubuntu6.3 (/var/lib/apt/lists/old-releases.ubuntu.com_ubuntu_dists_jaunty-updates_main_binary-amd64_Packages) (/var/lib/apt/lists/old-releases.ubuntu.com_ubuntu_dists_jaunty-security_main_binary-amd64_Packages) (/var/lib/dpkg/status)
     Description Language: 
                     File: /var/lib/apt/lists/old-releases.ubuntu.com_ubuntu_dists_jaunty-updates_main_binary-amd64_Packages
                      MD5: 1bbdc717d9acdb44db940928d570e749

    2.9-4ubuntu6 (/var/lib/apt/lists/old-releases.ubuntu.com_ubuntu_dists_jaunty_main_binary-amd64_Packages)
     Description Language: 
                     File: /var/lib/apt/lists/old-releases.ubuntu.com_ubuntu_dists_jaunty_main_binary-amd64_Packages
                      MD5: 1bbdc717d9acdb44db940928d570e749


    Reverse Depends: 
      gcc-3.4,libc6-dev 2.3.5-10
      virtualbox-ose-source,libc6-dev
      virtualbox-ose-guest-source,libc6-dev
      libwxgtk2.6-dev,libc6-dev
      ...
      comerr-dev,libc6-dev
      build-essential,libc6-dev
    Dependencies: 
    2.9-4ubuntu6.3 - libc6 (5 2.9-4ubuntu6.3) linux-libc-dev (0 (null)) glibc-doc (0 (null)) manpages-dev (0 (null)) gcc (16 (null)) c-compiler (0 (null)) libstdc++2.10-dev (3 1:2.95.2-15) gcc-2.95 (3 1:2.95.3-8) binutils (3 2.17cvs20070426-1) libc-dev (0 (null)) man-db (1 2.3.10-41) gettext (1 0.10.26-1) ppp (1 2.2.0f-24) libgdbmg1-dev (1 1.7.3-24) 
    2.9-4ubuntu6 - libc6 (5 2.9-4ubuntu6) linux-libc-dev (0 (null)) glibc-doc (0 (null)) manpages-dev (0 (null)) gcc (16 (null)) c-compiler (0 (null)) libstdc++2.10-dev (3 1:2.95.2-15) gcc-2.95 (3 1:2.95.3-8) binutils (3 2.17cvs20070426-1) libc-dev (0 (null)) man-db (1 2.3.10-41) gettext (1 0.10.26-1) ppp (1 2.2.0f-24) libgdbmg1-dev (1 1.7.3-24) 
    Provides: 
    2.9-4ubuntu6.3 - libc-dev 
    2.9-4ubuntu6 - libc-dev 
    Reverse Provides: 




simulate libc6 downgrades
---------------------------

sudo apt-cache showpkg libc6 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    aracity@aracity-desktop:~$ sudo apt-cache showpkg libc6    

    Package: libc6
    Versions: 
    2.12.1-0ubuntu10.2 (/var/lib/dpkg/status)
     Description Language: 
                     File: /var/lib/dpkg/status
                      MD5: 5089b4da6684d7432ab618fb5b79cec5

    2.9-4ubuntu6.3 (/var/lib/apt/lists/old-releases.ubuntu.com_ubuntu_dists_jaunty-updates_main_binary-amd64_Packages) (/var/lib/apt/lists/old-releases.ubuntu.com_ubuntu_dists_jaunty-security_main_binary-amd64_Packages)
     Description Language: 
                     File: /var/lib/apt/lists/old-releases.ubuntu.com_ubuntu_dists_jaunty-updates_main_binary-amd64_Packages
                      MD5: fc3001b0b90a1c8e6690b283a619d57f

    2.9-4ubuntu6 (/var/lib/apt/lists/old-releases.ubuntu.com_ubuntu_dists_jaunty_main_binary-amd64_Packages)
     Description Language: 
                     File: /var/lib/apt/lists/old-releases.ubuntu.com_ubuntu_dists_jaunty_main_binary-amd64_Packages
                      MD5: fc3001b0b90a1c8e6690b283a619d57f


    Reverse Depends: 
      g77-3.4,libc6 2.7-1
      libc-bin,libc6
      libc-bin,libc6 2.10
      ...

      acct,libc6 2.4
      abiword-plugin-mathview,libc6 2.4
      abiword-plugin-grammar,libc6 2.2.5
      abiword,libc6 2.7
    Dependencies: 
    2.12.1-0ubuntu10.2 - libc-bin (5 2.12.1-0ubuntu10.2) libgcc1 (0 (null)) tzdata (0 (null)) findutils (2 4.4.0-2ubuntu2) glibc-doc (0 (null)) debconf (16 (null)) debconf-2.0 (0 (null)) locales (0 (null)) belocs-locales-bin (0 (null)) tzdata (3 2007k-1) tzdata-etch (0 (null)) nscd (3 2.12) belocs-locales-bin (0 (null)) 
    2.9-4ubuntu6.3 - libgcc1 (0 (null)) findutils (2 4.4.0-2ubuntu2) locales (0 (null)) glibc-doc (0 (null)) libterm-readline-gnu-perl (3 1.15-2) tzdata (3 2007k-1) tzdata-etch (0 (null)) nscd (3 2.9) belocs-locales-bin (0 (null)) 
    2.9-4ubuntu6 - libgcc1 (0 (null)) findutils (2 4.4.0-2ubuntu2) locales (0 (null)) glibc-doc (0 (null)) libterm-readline-gnu-perl (3 1.15-2) tzdata (3 2007k-1) tzdata-etch (0 (null)) nscd (3 2.9) belocs-locales-bin (0 (null)) 
    Provides: 
    2.12.1-0ubuntu10.2 - glibc-2.11-1 
    2.9-4ubuntu6.3 - glibc-2.9-1 
    2.9-4ubuntu6 - glibc-2.9-1 
    Reverse Provides: 


sudo apt-get -s install libc6=2.9-4ubuntu6.3
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This downgrade satisfies libc6-dev

::

    aracity@aracity-desktop:~$ sudo apt-get -s install libc6=2.9-4ubuntu6.3
    Reading package lists... Done
    Building dependency tree       
    Reading state information... Done
    You might want to run `apt-get -f install' to correct these:
    The following packages have unmet dependencies:
      libc-bin: Breaks: libc6 (< 2.10) but 2.9-4ubuntu6.3 is to be installed
      libgcc1: Depends: gcc-4.5-base (= 4.5.1-7ubuntu2) but it is not installable
    E: Unmet dependencies. Try 'apt-get -f install' with no packages (or specify a solution).


sudo apt-get -s install libc6=2.9-4ubuntu6 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    aracity@aracity-desktop:~$ sudo apt-get -s install libc6=2.9-4ubuntu6  
    Reading package lists... Done
    Building dependency tree       
    Reading state information... Done
    You might want to run `apt-get -f install' to correct these:
    The following packages have unmet dependencies:
      libc-bin: Breaks: libc6 (< 2.10) but 2.9-4ubuntu6 is to be installed
      libc6-dev: Depends: libc6 (= 2.9-4ubuntu6.3) but 2.9-4ubuntu6 is to be installed
      libgcc1: Depends: gcc-4.5-base (= 4.5.1-7ubuntu2) but it is not installable
    E: Unmet dependencies. Try 'apt-get -f install' with no packages (or specify a solution).


sudo dpkg -s libc-bin      
~~~~~~~~~~~~~~~~~~~~~~~~

::

    aracity@aracity-desktop:~$ sudo dpkg -s libc-bin                     
    Package: libc-bin
    Status: install ok installed
    Priority: required
    Section: libs
    Installed-Size: 1796
    Maintainer: Ubuntu Core developers <ubuntu-devel-discuss@lists.ubuntu.com>
    Architecture: amd64
    Source: eglibc
    Version: 2.12.1-0ubuntu10.2
    Replaces: libc0.1, libc0.3, libc6, libc6.1
    Breaks: libc0.1 (<< 2.10), libc0.3 (<< 2.10), libc6 (<< 2.10), libc6.1 (<< 2.10)
    Conffiles:
     /etc/bindresvport.blacklist 154db0e55fa99051ff1bd99e5b2c0584
     /etc/gai.conf 629c0e2f8276b26c29b95f7ed53074d7
     /etc/ld.so.conf.d/libc.conf d4d833fd095fb7b90e1bb4a547f16de6
    Description: Embedded GNU C Library: Binaries
     This package contains utility programs related to the GNU C Library.
     .
      * catchsegv: catch segmentation faults in programs
      * getconf: query system configuration variables
      * getent: get entries from administrative databases
      * iconv, iconvconfig: convert between character encodings
      * ldd, ldconfig: print/configure shared library dependencies
      * locale, localedef: show/generate locale definitions
      * rpcinfo: report RPC information
      * tzselect, zdump, zic: select/dump/compile time zones
    Homepage: http://www.eglibc.org
    Original-Maintainer: GNU Libc Maintainers <debian-glibc@lists.debian.org>


sudo apt-cache showpkg libc-bin
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    aracity@aracity-desktop:~$ sudo apt-cache showpkg libc-bin   
    Package: libc-bin
    Versions: 
    2.12.1-0ubuntu10.2 (/var/lib/dpkg/status)
     Description Language: 
                     File: /var/lib/dpkg/status
                      MD5: 90523935f9c97be3a67bc8a1596de79b


    Reverse Depends: 
      libc6,libc-bin 2.12.1-0ubuntu10.2
    Dependencies: 
    2.12.1-0ubuntu10.2 - libc0.1 (3 2.10) libc0.3 (3 2.10) libc6 (3 2.10) libc6.1 (3 2.10) libc0.1 (0 (null)) libc0.3 (0 (null)) libc6 (0 (null)) libc6.1 (0 (null)) 
    Provides:     2.12.1-0ubuntu10.2 - 
    Reverse Provides: 



holding
----------

* http://ilostmynotes.blogspot.tw/2012/11/solved-dependency-problems-prevent.html


Put pkg on hold, then check the fix.::

    aracity@aracity-desktop:~$ echo "libc6-dev hold" | sudo dpkg --set-selections
    aracity@aracity-desktop:~$ sudo dpkg --get-selections
    acl                                             install
    acpi-support                                    install
    ...
    aracity@aracity-desktop:~$ sudo dpkg --get-selections libc6-dev
    libc6-dev                                       hold



::

    aracity@aracity-desktop:~$ sudo apt-cache policy libc6
    libc6:
      Installed: 2.12.1-0ubuntu10.2
      Candidate: 2.12.1-0ubuntu10.2
      Version table:
     *** 2.12.1-0ubuntu10.2 0
            100 /var/lib/dpkg/status
         2.9-4ubuntu6.3 0
            500 http://old-releases.ubuntu.com jaunty-updates/main Packages
            500 http://old-releases.ubuntu.com jaunty-security/main Packages
         2.9-4ubuntu6 0
            500 http://old-releases.ubuntu.com jaunty/main Packages

::

    aracity@aracity-desktop:~$ sudo apt-cache policy libc6-dev
    libc6-dev:
      Installed: 2.9-4ubuntu6.3
      Candidate: 2.9-4ubuntu6.3
      Version table:
     *** 2.9-4ubuntu6.3 0
            500 http://old-releases.ubuntu.com jaunty-updates/main Packages
            500 http://old-releases.ubuntu.com jaunty-security/main Packages
            100 /var/lib/dpkg/status
         2.9-4ubuntu6 0
            500 http://old-releases.ubuntu.com jaunty/main Packages


::

    aracity@aracity-desktop:~$ sudo apt-cache policy libgcc1  
    libgcc1:
      Installed: 1:4.5.1-7ubuntu2
      Candidate: 1:4.5.1-7ubuntu2
      Version table:
     *** 1:4.5.1-7ubuntu2 0
            100 /var/lib/dpkg/status
         1:4.3.3-5ubuntu4 0
            500 http://old-releases.ubuntu.com jaunty/main Packages
    aracity@aracity-desktop:~$ 



