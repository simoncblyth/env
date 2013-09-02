Ubuntu Package Management
=========================

References
-----------

* https://help.ubuntu.com/community/InstallingSoftware


Tools 
------

* dpkg
* apt-get
* apt-cache


dpkg or apt-get
~~~~~~~~~~~~~~~~

* http://debian-handbook.info/browse/wheezy/sect.manipulating-packages-with-dpkg.html

`dpkg` should be seen as a system tool (backend), and `apt-get` as a tool closer to
the user, which overcomes the limitations of the former. These tools work
together, each one with its particularities, suited to specific tasks.


List installed packages
------------------------

grep /var/lib/dpkg/status
~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    aracity@aracity-desktop:~$ grep Package: /var/lib/dpkg/status | sort | head -10 
    Package: acl
    Package: acpi-support
    Package: acpid
    Package: adduser
    Package: alacarte
    Package: alsa-base
    Package: alsa-utils
    Package: anacron
    Package: app-install-data
    Package: app-install-data-partner


dpkg-query
~~~~~~~~~~~

::

    aracity@aracity-desktop:~$ dpkg-query -Wf '${Installed-Size}\t${Package}\n' | sort -n - | tail -10
    40772   evolution-common
    42052   gnome-games-data
    44860   openoffice.org-common
    59784   ubuntu-docs
    65072   linux-headers-2.6.28-19
    74940   libroot5.18
    109552  linux-image-2.6.28-11-generic
    109956  linux-image-2.6.28-18-generic
    109972  linux-image-2.6.28-19-generic
    119564  openoffice.org-core
    aracity@aracity-desktop:~$ 


Search for packages
--------------------

apt-cache search
~~~~~~~~~~~~~~~~~

::

    aracity@aracity-desktop:~$ apt-cache search subversion
    ...
    python-subversion-dbg - Python bindings for Subversion (debug extension)
    subversion - Advanced version control system
    subversion-tools - Assorted tools related to Subversion
    gforge-plugin-scmsvn - collaborative development tool - Subversion plugin
    ikiwiki - a wiki compiler
    libapache2-svn - Subversion server modules for Apache
    libsvn-java - Java bindings for Subversion
    libsvn-ruby - Ruby bindings for Subversion (dummy package)
    libsvn-ruby1.8 - Ruby bindings for Subversion


Info for a package
-----------------------


understanding unmet dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    The following packages have unmet dependencies:
      libc6-dev: Depends: libc6 (= 2.9-4ubuntu6.3) but 2.12.1-0ubuntu10.2 is to be installed
      libgcc1: Depends: gcc-4.5-base (= 4.5.1-7ubuntu2) but it is not installable
 



dpkg --status OR dpkg -s
~~~~~~~~~~~~~~~~~~~~~~~~~~~

* :google:`dpkg -s Status field`




::

    aracity@aracity-desktop:~$ dpkg --status libc6
    Package: libc6
    Status: install ok unpacked
    Priority: required
    Section: libs
    Installed-Size: 10232
    Maintainer: Ubuntu Core developers <ubuntu-devel-discuss@lists.ubuntu.com>
    Architecture: amd64
    Source: eglibc
    Version: 2.12.1-0ubuntu10.2
    Config-Version: 2.9-4ubuntu6.3
    Replaces: belocs-locales-bin
    Provides: glibc-2.11-1
    Depends: libc-bin (= 2.12.1-0ubuntu10.2), libgcc1, tzdata, findutils (>= 4.4.0-2ubuntu2)
    Suggests: glibc-doc, debconf | debconf-2.0, locales
    Breaks: nscd (<< 2.12)
    Conflicts: belocs-locales-bin, tzdata (<< 2007k-1), tzdata-etch
    Conffiles:
     /etc/ld.so.conf.d/x86_64-linux-gnu.conf 593ad12389ab2b6f952e7ede67b8fbbf
     /etc/bindresvport.blacklist db84c47f31f8d5a334a4053d8368e902 obsolete
     /etc/gai.conf ab538b366edabe44a9a4020fdd3d93a4 obsolete
     /etc/ld.so.conf.d/libc.conf d4d833fd095fb7b90e1bb4a547f16de6 obsolete
    Description: Embedded GNU C Library: Shared libraries
     Contains the standard libraries that are used by nearly all programs on
     the system. This package includes shared versions of the standard C library
     and the standard math library, as well as many others.
    Homepage: http://www.eglibc.org
    Original-Maintainer: GNU Libc Maintainers <debian-glibc@lists.debian.org>



::

    aracity@aracity-desktop:~$ dpkg --status libc6-dev
    Package: libc6-dev
    Status: install ok installed
    Priority: optional
    Section: libdevel
    Installed-Size: 11456
    Maintainer: Ubuntu Core developers <ubuntu-devel-discuss@lists.ubuntu.com>
    Architecture: amd64
    Source: glibc
    Version: 2.9-4ubuntu6.3
    Replaces: man-db (<= 2.3.10-41), gettext (<= 0.10.26-1), ppp (<= 2.2.0f-24), libgdbmg1-dev (<= 1.7.3-24)
    Provides: libc-dev
    Depends: libc6 (= 2.9-4ubuntu6.3), linux-libc-dev
    Recommends: gcc | c-compiler
    Suggests: glibc-doc, manpages-dev
    Conflicts: libstdc++2.10-dev (<< 1:2.95.2-15), gcc-2.95 (<< 1:2.95.3-8), binutils (<< 2.17cvs20070426-1), libc-dev
    Description: GNU C Library: Development Libraries and Header Files
     Contains the symlinks, headers, and object files needed to compile
     and link programs which use the standard C library.
    Original-Maintainer: GNU Libc Maintainers <debian-glibc@lists.debian.org>




apt-cache showpkg
~~~~~~~~~~~~~~~~~~

::

    aracity@aracity-desktop:~$ apt-cache showpkg subversion
    Package: subversion
    Versions: 
    1.5.4dfsg1-1ubuntu2.1 (/var/lib/apt/lists/tw.archive.ubuntu.com_ubuntu_dists_jaunty-updates_main_binary-amd64_Packages) (/var/lib/apt/lists/security.ubuntu.com_ubuntu_dists_jaunty-security_main_binary-amd64_Packages)
     Description Language: 
                     File: /var/lib/apt/lists/tw.archive.ubuntu.com_ubuntu_dists_jaunty-updates_main_binary-amd64_Packages
                      MD5: 15da1bb51fb2e9ea5e25b3a489b864d9

    1.5.4dfsg1-1ubuntu2 (/var/lib/apt/lists/tw.archive.ubuntu.com_ubuntu_dists_jaunty_main_binary-amd64_Packages)
     Description Language: 
                     File: /var/lib/apt/lists/tw.archive.ubuntu.com_ubuntu_dists_jaunty_main_binary-amd64_Packages
                      MD5: 15da1bb51fb2e9ea5e25b3a489b864d9

    ...




List Files in a package
------------------------

dpkg -L
~~~~~~~~~

::

    aracity@aracity-desktop:~$ dpkg -L libc6 | head -10
    /.
    /etc
    /etc/ld.so.conf.d
    /etc/ld.so.conf.d/x86_64-linux-gnu.conf
    /lib
    /lib/libanl-2.12.1.so
    /lib/libBrokenLocale-2.12.1.so
    /lib/librt-2.12.1.so
    /lib/libpcprofile.so
    /lib/libc-2.12.1.so 






Install packages
----------------

apt-get
~~~~~~~~~

