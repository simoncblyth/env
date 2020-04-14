bash-vi(){ vi $BASH_SOURCE ; }

bash-env(){
  elocal-
}

bash-usage(){ cat << EOU

bash
=====


macOS bash is ancient, due to GPLv3 licencing of 3.2+
------------------------------------------------------

* https://itnext.io/upgrading-bash-on-macos-7138bd1066ba

::

    epsilon:~ blyth$ echo $BASH_VERSION
    3.2.57(1)-release


::

    sudo port selfupdate
    sudo port install bash
    sudo vim /etc/shells   # adding /opt/local/bin/bash

    chsh -s /opt/local/bin/bash

::

    epsilon:~ blyth$ chsh -s /opt/local/bin/bash
    Changing shell for blyth.
    Password for blyth: 
    epsilon:~ blyth$ 


.bashrc or .bash_profile
-------------------------

* http://www.joshstaiger.org/archives/2005/07/bash_profile_vs.html

.bash_profile
    on login 

.bashrc
    every new terminal window


macOS Terminal.app exception
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An exception to the terminal window guidelines is Mac OS X’s Terminal.app,
which runs a login shell by default for each new terminal window, calling
.bash_profile instead of .bashrc. Other GUI terminal emulators may do the same,
but most tend not to.

Recommendation
~~~~~~~~~~~~~~~~

Most of the time you don’t want to maintain two separate config files for login
and non-login shells — when you set a PATH, you want it to apply to both. You
can fix this by sourcing .bashrc from your .bash_profile file, then putting
PATH and common settings in .bashrc.

To do this, add the following lines to .bash_profile:
if [ -f ~/.bashrc ]; then
   source ~/.bashrc
fi
Now when you login to your machine from a console .bashrc will be called.


Debugging Tip For Disappearing Functions
-----------------------------------------

When a function disappears (this happened with glfw--)
you probably have a stray space on an end token::

    cat << EON > /dev/null

    Note line 
    Note line 
    Note line 

    EON**

Stray spaces after the EON can easily cause functions to disappear.
To find the problem in a large file move a probe function around that 
you keep changing the output from to different places in the
file to identify where the rot sets in. 

::

    glfw-check(){ echo TEST5 ; }



Redirect bug in bash 3.2
---------------------------

* https://stackoverflow.com/questions/1279953/how-to-execute-the-output-of-a-command-within-the-current-shell

::

   $ ls | sed ... | source /dev/stdin

UPDATE: This works in bash 4.0, as well as tcsh, and dash (if you change source to .). 
Apparently this was buggy in bash 3.2. From the bash 4.0 release notes::

    Fixed a bug that caused `.' to fail to read and execute commands 
    from non-regular files such as devices or named pipes.

Ref
----

http://www.gnu.org/software/bash/manual/bashref.html#Invoking-Bash

EOU
}

bash-dir(){
   echo $(dirname $BASH_SOURCE)
}

bash-source(){
   echo $BASH_SOURCE
}


bash-funcdef(){
  local dir=$1
  local def="function func-dir(){ echo $dir ; }"
  echo $def
}


bash-positional-args(){

  local msg="=== $FUNCNAME :"
  echo $msg initially $* 
  local args=$* 
  set -- 
  echo $msg after set $* 
  set $args 
  echo $msg after reset $* 


}


bash-create-func-with-a-func(){

   local def=$(bash-funcdef $(dirname $0))
   echo $def
   eval $def

}


bash-slash-count(){
   local s=$1
   local ifs=$IFS
   IFS=/
   bash-nargs $s 
   IFS=$ifs
   
}

bash-nargs(){
   echo $# 
}



bash-getopts-wierdness(){


cat <<  EOW

In a fresh shell this works once only ....

simon:base blyth$ . bash.bash  
simon:base blyth$ bash-getopts -r -x red greed blu
OPTFIND
after opt parsing red greed blu
dummy=-x
rebuild=-r


simon:base blyth$ bash-getopts -r -x red greed blu
OPTFIND
after opt parsing red greed blu
dummy=
rebuild=

EOW

}


bash-heredoc(){

  cat << EOS

Backticks do get expanded 

  uname :  `uname`
  date  :  `date`

EOS

}


bash-heredoc-quoted(){

  cat << 'EOS'

Backticks do NOT get expanded when quote the end token

  uname :  `uname`
  date  :  `date`

EOS

}





bash-getopts(){

   # http://www.linux.com/articles/113836

   #
   #  The options must come first ...
   #       bash-getopts -r red green blue
   # 
   #  otherwise they are ignored 
   #        bash-getopts red green blue -r 
   #


   echo raw args \$@:$@  \$*:$* 


   local rebuild=""
   local dummy=""


   ## leading colon causes error messages not ro be skipped
   local o
   while getopts "rx" o ; do      
      case $o in
        r) rebuild="-r";;
        x) dummy="-x" ;;
      esac
   done



   echo OPTFIND $OPTFIND
   shift $((${OPTIND}-1))

   env | grep OPT

   echo after opt parsing   $@
   local
}




#echo BASH_SOURCE $BASH_SOURCE 
