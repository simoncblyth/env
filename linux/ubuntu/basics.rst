Ubuntu Basics
====================

* http://superuser.com/questions/154333/what-is-the-difference-between-debian-and-ubuntu
* https://wiki.ubuntu.com/Ubuntu/ForDebianDevelopers

Checking Ubuntu Version
--------------------------

* https://help.ubuntu.com/community/CheckingYourUbuntuVersion

lsb_release
~~~~~~~~~~~

* (Chroma recommends Ubuntu 11.04)

::

    aracity@aracity-desktop:~$ lsb_release -a
    No LSB modules are available.
    Distributor ID: Ubuntu
    Description:    Ubuntu 9.04
    Release:        9.04
    Codename:       jaunty


Ubuntu Lifecycle
-----------------

Ubuntu releases die after 18 months, with backing repositories going offline, 
so `apt-get update` will stop working and will not be able to install updates.

* http://www.canonical.com/enterprise-services/support/server/support-life-cycles


Diagram of lifecycles.  

* https://wiki.ubuntu.com/LTS



After death Ubuntu
--------------------


* https://help.ubuntu.com/community/UpgradeNotes
* https://help.ubuntu.com/community/EOLUpgrades/Jaunty


Update sources.list (30 Aug 2013)
------------------------------------

* http://askubuntu.com/questions/48025/what-to-do-when-cant-update-anymore-with-apt-get

Edit "/etc/apt/sources.list" (with root permissions) substituting all the links: "http://archive.ubuntu.com/..." 
for "http://old-releases.ubuntu.com/..."


Reconnecting to a backing old-releases repository, in order for `apt-get` to work again.::

    aracity@aracity-desktop:/etc/apt$ sudo cp sources.list sources.list.apr15-2010
    aracity@aracity-desktop:/etc/apt$ sudo cp sources.list sources.list.new
    aracity@aracity-desktop:/etc/apt$ sudo perl -pi -e 's,tw.archive.ubuntu.com,old-releases.ubuntu.com,g' sources.list.new
    aracity@aracity-desktop:/etc/apt$ sudo perl -pi -e 's,security.ubuntu.com,old-releases.ubuntu.com,g' sources.list.new
    aracity@aracity-desktop:/etc/apt$ diff sources.list sources.list.new
    5,6c5,6
    < deb http://tw.archive.ubuntu.com/ubuntu/ jaunty main restricted
    < deb-src http://tw.archive.ubuntu.com/ubuntu/ jaunty main restricted
    ---
    > deb http://old-releases.ubuntu.com/ubuntu/ jaunty main restricted
    > deb-src http://old-releases.ubuntu.com/ubuntu/ jaunty main restricted
    10,11c10,11
    aracity@aracity-desktop:/etc/apt$ 


::

    aracity@aracity-desktop:~$ sudo apt-get update 
    ...
    Fetched 10.7MB in 11s (928kB/s)
    Reading package lists... Done
    aracity@aracity-desktop:~$ 

