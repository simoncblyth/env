linux-env(){ echo -n ; }
linux-vi(){ vi $BASH_SOURCE ; }
linux-usage(){ cat << EOU
alma linux
==========


reset root pw
--------------

https://glesys.se/kb/artikel/how-to-reset-your-root-password-in-almalinux-9


grant sudoer to user
---------------------

* https://coderstalk.blogspot.com/2023/06/how-to-add-sudo-user-on-almalinux-92.html

This didnt work::

    usermod -aG wheel blyth

Editing the /etc/sudoers did

* https://bobcares.com/blog/almalinux-add-user-to-sudoers/



EOU
}
