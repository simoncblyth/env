Non-Root package management
=============================

Package Management convenience without root access, can it be done ?

* :google:`non-root yum`
* :google:`non-root package management yum rpm`

* http://yum.baseurl.org/wiki/RunningWithoutRoot
* http://unix.stackexchange.com/questions/5535/non-root-package-managers


RPM background
----------------

* https://docs.fedoraproject.org/en-US/Fedora_Draft_Documentation/0.1/html/RPM_Guide/index.html


RPM configuration via release pkg
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The equivalent of `redhat-release` package on ordinary redhat appears to be `sl-release` on Scientific Linux machines.

::

    [blyth@belle7 ~]$ sudo yum --enablerepo=epel search sl-release
    sl-release.i386 : Scientific Linux release file
    sl-release-notes.noarch : Scientific Linux release notes files

::

    [blyth@belle7 ~]$ rpm -ql sl-release
    /etc/issue
    /etc/issue.net
    /etc/pki/rpm-gpg
    /etc/pki/rpm-gpg/RPM-GPG-KEY
    /etc/pki/rpm-gpg/RPM-GPG-KEY-CentOS-5
    /etc/pki/rpm-gpg/RPM-GPG-KEY-adobe-linux
    /etc/pki/rpm-gpg/RPM-GPG-KEY-atrpms
    /etc/pki/rpm-gpg/RPM-GPG-KEY-beta
    /etc/pki/rpm-gpg/RPM-GPG-KEY-centos4
    /etc/pki/rpm-gpg/RPM-GPG-KEY-cern
    /etc/pki/rpm-gpg/RPM-GPG-KEY-csieh
    /etc/pki/rpm-gpg/RPM-GPG-KEY-dag
    /etc/pki/rpm-gpg/RPM-GPG-KEY-dawson
    /etc/pki/rpm-gpg/RPM-GPG-KEY-dries
    /etc/pki/rpm-gpg/RPM-GPG-KEY-fedora
    /etc/pki/rpm-gpg/RPM-GPG-KEY-fedora-test
    /etc/pki/rpm-gpg/RPM-GPG-KEY-jpolok
    /etc/pki/rpm-gpg/RPM-GPG-KEY-redhat
    /etc/pki/rpm-gpg/RPM-GPG-KEY-redhat-rhx
    /etc/pki/rpm-gpg/RPM-GPG-KEY-wiesand
    /etc/redhat-release
    /etc/sysconfig/rhn
    /etc/sysconfig/rhn/sources
    /usr/share/doc/sl-release-5.1
    /usr/share/doc/sl-release-5.1/GPL
    /usr/share/doc/sl-release-5.1/autorun-template
    /usr/share/sl-release/default-bookmarks.html


chroot jail
-------------

* :google:`yum chroot jail example`
* http://geek.co.il/2010/03/14/how-to-build-a-chroot-jail-environment-for-centos
* http://prefetch.net/articles/yumchrootlinux.html


`mock -r configname`
-----------------------

* http://fedoraproject.org/wiki/Projects/Mock
* https://fedoraproject.org/wiki/Using_Mock_to_test_package_builds#Using_mock_as_a_chroot_sandbox_tool
* http://docs.fedoraproject.org/en-US/Fedora_Draft_Documentation/0.1/html-single/Packagers_Guide/index.html

Mock can be used to create chroots for testing things, not just building packages. 
Create a config file that points to the repo(s) of your choice, where your test packages are

::

    mock -r <config-name> --init
    mock -r <config-name> --install <your packages>
    mock -r <config-name> --shell

Why use mock to shell, why not chroot directly? 

   * Using mock to "shell" will allow mock to create the mountpoints you'll
     probably need inside the chroot. 


Nix
----

* http://nixos.org/nix/


Zero Install
--------------

* http://0install.net/



