# === func-gen- : linux/gnome/gnome fgp linux/gnome/gnome.bash fgn gnome fgh linux/gnome
gnome-src(){      echo linux/gnome/gnome.bash ; }
gnome-source(){   echo ${BASH_SOURCE:-$(env-home)/$(gnome-src)} ; }
gnome-vi(){       vi $(gnome-source) ; }
gnome-env(){      elocal- ; }
gnome-usage(){ cat << EOU

GNOME : Linux Desktop
=======================


Restart from a freeze
------------------------

* https://www.addictivetips.com/ubuntu-linux-tips/three-ways-to-restart-gnome-without-rebooting-system/

Save all your open files, commit to repos etc,  then either:

1. press Ctrl–Alt–Backspace.
2. sudo /etc/init.d/gdm restart
3. sudo killall gnome-panel



EOU
}
gnome-dir(){ echo $(local-base)/env/linux/gnome/linux/gnome-gnome ; }
gnome-cd(){  cd $(gnome-dir); }
gnome-mate(){ mate $(gnome-dir) ; }
gnome-get(){
   local dir=$(dirname $(gnome-dir)) &&  mkdir -p $dir && cd $dir

}
