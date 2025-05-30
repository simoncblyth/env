# === func-gen- : linux/gnome/gnome fgp linux/gnome/gnome.bash fgn gnome fgh linux/gnome
gnome-src(){      echo linux/gnome/gnome.bash ; }
gnome-source(){   echo ${BASH_SOURCE:-$(env-home)/$(gnome-src)} ; }
gnome-vi(){       vi $(gnome-source) ; }
gnome-env(){      elocal- ; }
gnome-usage(){ cat << EOU

GNOME : Linux Desktop
=======================


Open terminal windows on workstation from ssh session
------------------------------------------------------

A::

    A[blyth@localhost ~]$ VIP=none gnome-terminal
    A[blyth@localhost ~]$ VIP=conda_ok gnome-terminal


HMM: but maybe GUI issue with these ?


::

    A[blyth@localhost ~]$ gnome-terminal --command="/bin/bash -i -c 'export VIP=none; gnome-terminal'"
    # Option “--command” is deprecated and might be removed in a later version of gnome-terminal.
    # Use “-- ” to terminate the options and put the command line to execute after it.
    A[blyth@localhost ~]$ 
    A[blyth@localhost ~]$ gnome-terminal -- /bin/bash -i -c 'export VIP=none; gnome-terminal'
    A[blyth@localhost ~]$ 
    A[blyth@localhost ~]$ gnome-terminal -- /bin/bash -i -c 'export VIP=conda_ok; gnome-terminal'
    A[blyth@localhost ~]$ gnome-terminal -- /bin/bash -i -c 'VIP=conda_ok gnome-terminal'


Probably missing DISPLAY, the below starting to work::

    A[blyth@localhost ~]$ env DISPLAY=:0 VIP=conda_ok gnome-terminal




Add custom keyboard shortcut 
----------------------------

* Power > Settings : Keyboard > Keyboard Shortcuts, Customize Shortcuts > Custom Shortcuts [+] and Enter:

  * Name eg [GnomeTerminal]
  * Command eg [gnome-terminal]
  * Shortcut eg [Ctrl+Alt+T] 



Issue : Frozen GUI 
---------------------

After using obs- find that gnome GUI is stuck, can enter text in terminals, 
and can connect via ssh. But cannot move windows around.   

Fix : kill the gnome-shell
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It comes back automatically with interactivity regained.

::

    ps aux | grep gnome   

    blyth      2628  9.5  0.7 7279536 478844 ?      Sl   Jul17 147:08 /usr/bin/gnome-shell 
    ## reveals /usr/bin/gnome-shell as the one taking the CPU cycles

    sudo kill -9 2628



EOU
}
gnome-dir(){ echo $(local-base)/env/linux/gnome/linux/gnome-gnome ; }
gnome-cd(){  cd $(gnome-dir); }
gnome-mate(){ mate $(gnome-dir) ; }
gnome-get(){
   local dir=$(dirname $(gnome-dir)) &&  mkdir -p $dir && cd $dir

}
