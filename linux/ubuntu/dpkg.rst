dpkg
=====

The `dpkg` command and the database of `.deb` files that is manages 
are the underpinning for package installation frontend commands such as 

* `apt-get` OR `dselect`   (choose one `apt-get` 
* `capt`

Extracts from below sources that are useful for dependency debugging.

* `man dpkg` 
* `man dpkg-query`

* http://www.infodrom.org/Debian/doc/maint/Maintenance-pkgmaint.html


frontends : `apt-get` OR `dselect`
-------------------------------------

* http://unix.stackexchange.com/questions/6002/what-are-pros-cons-of-dselect-and-apt-get

  * `apt-get` is favored over `dselect` apparently



package state
--------------

From `man dpkg`.

`dpkg` maintains some usable information about available packages. 
The information is divided in three classes: `states`, `selection states` and `flags`. 
These values are intended to be changed mainly with `dselect`.

states
~~~~~~~~

not-installed
              The package is not installed on your system.

config-files
              Only the configuration files of the package exist on the system.

half-installed
              The installation of the package has been started, but not completed for some reason.

unpacked
              The package is unpacked, but not configured.

half-configured
              The package is unpacked and configuration has been started, but not yet completed for some reason.

triggers-awaited
              The package awaits trigger processing by another package.

triggers-pending
              The package has been triggered.

installed
              The package is unpacked and configured OK.


selection states
~~~~~~~~~~~~~~~~~~



flags
~~~~~~



`dpkg -s, --status package-name`
---------------------------------

* :google:`debian package status database`

Report status of specified package. This just displays the entry in the installed package status database.
To just see package metadata, without local installation infomation, use `apt-cache show`.


Version
~~~~~~~~~

The version number of a package. The format is: `[epoch:]upstream_version[-debian_revision]`



::

    aracity@aracity-desktop:~$ dpkg -s libc6 | grep ^Version
    Version: 2.12.1-0ubuntu10.2
    aracity@aracity-desktop:~$ dpkg -s libc6-dev | grep ^Version
    Version: 2.9-4ubuntu6.3


Config-Version
~~~~~~~~~~~~~~~

If a package is not installed or not configured, this field in dpkg's status
file records the last version of the package which was successfully configured.

* so presence of this field indicates problems 

::

    aracity@aracity-desktop:~$ dpkg -s libc6 | grep Config-Version
    Config-Version: 2.9-4ubuntu6.3
    aracity@aracity-desktop:~$ dpkg -s libc6-dev | grep Config-Version


Status field enumerations
~~~~~~~~~~~~~~~~~~~~~~~~~~

* http://www.fifi.org/doc/libapt-pkg-doc/dpkg-tech.html/ch1.html

::

    aracity@aracity-desktop:~$ dpkg --status libc6-dev | grep Status:
    Status: install ok installed
    aracity@aracity-desktop:~$ dpkg --status libc6 | grep Status:
    Status: install ok unpacked


The exact format for the "Status:" field is:  `Status: Want Flag Status`

Want
^^^^^

* unknown, 
* install, 
* hold, 
* deinstall, 
* purge. 

Flag
^^^^^

* ok, 
* reinstreq, 
* hold, 
* hold-reinstreq. 

Status
^^^^^^^

`not-installed`
         No files are installed from the package, it has no config files left, it uninstalled cleanly if it ever was installed.
`unpacked`
         The basic files have been unpacked (and are listed in `/var/lib/dpkg/info/[package].list`. 
         There are config files present, but the `postinst` script has **NOT** been run.
`half-configured`
         The package was `installed` and `unpacked`, but the `postinst` script failed in some way.
`installed`
         All files for the package are `installed`, and the configuration was also successful.
`half-installed`
         An attempt was made to remove the packagem but there was a failure in the prerm script.
`config-files`
         The package was "removed", not "purged". The config files are left, but nothing else.
`post-inst-failed`
         Old name for half-configured. Do not use.
`removal-failed`
         Old name for half-installed. Do not use.



`dpkg --set-selections`
-------------------------

Set package selections using file read from stdin. This file should be
in the format '<package> <state>', where state is one of install, hold,
deinstall or purge. Blank lines and comment lines beginning with '#' are also
permitted.

Example::

    echo "libc6-dev hold" | dpkg --set-selections
         

`dpkg -C, --audit`
-------------------

Searches for packages that have been installed only partially on your system.
dpkg will suggest what to do with them to get them working.


::

    aracity@aracity-desktop:~$ dpkg --audit 
    The following packages have been unpacked but not yet configured.
    They must be configured using dpkg --configure or the configure
    menu option in dselect for them to work:

     libgcc1              GCC support library
     libc6                Embedded GNU C Library: Shared libraries

    The following packages are only half configured, probably due to problems
    configuring them the first time.  The configuration should be retried using
    dpkg --configure <package> or the configure menu option in dselect:
     gracie               OpenID server for local PAM accounts




