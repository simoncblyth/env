
cms01 /data mount
==================


::

    [blyth@cms01 ~]$ df -h
    Filesystem            Size  Used Avail Use% Mounted on
    /dev/hda2              20G   17G  1.4G  93% /
    /dev/hda1             251M   28M  210M  12% /boot
    none                 1013M     0 1013M   0% /dev/shm
    /dev/hda5              77G   71G  2.5G  97% /home
    [blyth@cms01 ~]$ 
    [blyth@cms01 ~]$ 
    [blyth@cms01 ~]$ sudo mount /data
    Password:
    [blyth@cms01 ~]$ df -h
    Filesystem            Size  Used Avail Use% Mounted on
    /dev/hda2              20G   17G  1.4G  93% /
    /dev/hda1             251M   28M  210M  12% /boot
    none                 1013M     0 1013M   0% /dev/shm
    /dev/hda5              77G   71G  2.5G  97% /home
    /dev/hda6             132G  106G   20G  85% /data
    [blyth@cms01 ~]$ 


