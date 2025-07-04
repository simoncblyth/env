bash-vi(){ vi $BASH_SOURCE ; }

bash-env(){
  elocal-
}

bash-usage(){ cat << EOU

bash
=====



workstation console env vs ssh env
-------------------------------------- 

* https://askubuntu.com/questions/121073/why-bash-profile-is-not-getting-sourced-when-opening-a-terminal

* http://mywiki.wooledge.org/DotFiles

.bash_profile vs .bashrc
---------------------------

* https://apple.stackexchange.com/questions/51036/what-is-the-difference-between-bash-profile-and-bashrc

.bash_profile is executed for login shells, while .bashrc is executed for
interactive non-login shells.

When you login (type username and password) via console, either sitting at the
machine, or remotely via ssh: .bash_profile is executed to configure your shell
before the initial command prompt.

But, if you’ve already logged into your machine and open a new terminal window
(xterm) then .bashrc is executed before the window command prompt. .bashrc is
also run when you start a new bash instance by typing /bin/bash in a terminal.

On OS X, Terminal by default runs a login shell every time, so this is a little
different to most other systems, but you can configure that in the preferences.


What I do
~~~~~~~~~~~~

Remove most of the distinction by within the .bash_profile sourcing the .bashrc


uppercase : requires newer bash 
---------------------------------

::

    zeta:~ blyth$ c=a
    zeta:~ blyth$ echo ${c^}
    -bash: ${c^}: bad substitution

    zeta:home blyth$ echo $BASH_VERSION
    3.2.57(1)-release

In macOS Sequoia Terminal.app Settings change::

    Shells open with: Default login shell

To::

    Shells open with: command (complete path) /opt/local/bin/bash


Then::

    zeta:h blyth$ c=a
    zeta:h blyth$ echo ${c^}
    A

    zeta:h blyth$ echo $BASH_VERSION
    5.2.37(1)-release



declare -a arr = (1 2 3)
-------------------------

* advantage of "declare -a" is local scoping 


split string into array using IFS and "read -a"
-------------------------------------------------

::

   cegs=10:20:30:40
   IFS=: read -a cegs_arr <<< "$cegs"
   cegs_elem=${#cegs_arr[@]}

   case $cegs_elem in  
       4) echo $msg 4 element CXS_CEGS $CXS_CEGS ;; 
       7) echo $msg 7 element CXS_CEGS $CXS_CEGS ;; 
       *) echo $msg ERROR UNEXPECTED $cegs_elem element CXS_CEGS $CXS_CEGS && exit 1  ;;  
   esac

   # quotes on the in variable required due to bug fixed in bash 4.3 according to 
   # https://stackoverflow.com/questions/918886/how-do-i-split-a-string-on-a-delimiter-in-bash



trim extension, extract string after char
--------------------------------------------

::

    epsilon:env blyth$ echo $name
    clhep-2.4.6.2.tgz
    epsilon:env blyth$ echo ${name%.*}
    clhep-2.4.6.2

    epsilon:env blyth$ stem=clhep-2.4.6.2
    epsilon:env blyth$ echo ${stem%-*}   # string before - 
    clhep
    epsilon:env blyth$ echo ${stem#*-}   # string after -
    2.4.6.2

    epsilon:env blyth$ echo ${vers//[.]/_}
    2_4_6_2
    epsilon:env blyth$ echo ${vers//./_}
    2_4_6_2



bash arguments
----------------

* https://unix.stackexchange.com/questions/129072/whats-the-difference-between-and

::

    #!/bin/bash

    echo "Using \"\$*\":"
    for a in "$*"; do
        echo $a;
    done

    echo -e "\nUsing \$*:"
    for a in $*; do
        echo $a;
    done

    echo -e "\nUsing \"\$@\":"
    for a in "$@"; do
        echo $a;
    done

    echo -e "\nUsing \$@:"
    for a in $@; do
        echo $a;
    done              

    epsilon:env blyth$ args2.sh red green blue "cyan magenta" yellow black 
    Using "$*":
    red green blue cyan magenta yellow black

    Using $*:
    red
    green
    blue
    cyan
    magenta
    yellow
    black

    Using "$@":
    red
    green
    blue
    cyan magenta
    yellow
    black

    Using $@:
    red
    green
    blue
    cyan
    magenta
    yellow
    black




join
------

::

   joi-(){ shift && echo "$*" ; }
   join(){ IFS=$1 joi- $* ; }
   [ "$(join : red green blue)" == "red:green:blue" ] && echo Y || echo N 

   joi-(){ echo "$*" ; }
   join(){ IFS=$1 joi- ${*:2} ; }
   [ "$(join : red green blue)" == "red:green:blue" ] && echo Y || echo N 






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


change prompt color with PS1
------------------------------


* https://www.cyberciti.biz/faq/bash-shell-change-the-color-of-my-shell-prompt-under-linux-or-unix/

::

    epsilon:offline blyth$ echo $PS1
    \h:\W \u\$

    epsilon:offline blyth$ export PS1="\e[0;31m[\u@\h \W]\$ \e[m "
    [blyth@epsilon offline]$    ## red prompt 




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



Double Square Bracket
------------------------

* http://mywiki.wooledge.org/BashFAQ/031





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


bash-join(){ local IFS=$1 ; shift ; echo "$*" ; }

bash-pkg-config-path-flakey(){
  : returns a colon delimited PKG_CONFIG_PATH string derived from CMAKE_PREFIX_PATH   

  local ifs=$IFS 
  IFS=: 
  read -ra CPP <<< "$CMAKE_PREFIX_PATH"   # split
  local i
  local n=${#CPP[@]}
  local pfx
  local PCP=() 
  for i in ${!CPP[@]}; do
      pfx=${CPP[i]}
      if [ -d "$pfx/lib/pkgconfig" ]; then 
          PCP+=("$pfx/lib/pkgconfig")
      elif [ -d "$pfx/lib64/pkgconfig" ]; then 
          PCP+=("$pfx/lib64/pkgconfig")
      fi   
      : NB lib64 is ignored when there is a lib  
  done 
  bash-join : "${PCP[@]}" 

  # suspect the cause of flakiness was accidentally leaving IFS set to colon 
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

bash-parameter-expansion(){
   local var=${1:-HOME}
   echo ${!var}
}

bash-path-prepend(){
   local var=${1:-PATH}
   local dir=${2:-/tmp}
   local l=${3:-:}  # delim

   echo var $var dir $dir delim $delim 
   echo ${!var}
   
   [[ "$l${!var}$l" != *"$l${dir}$l"* ]] && eval $var=$dir$l${!var} 
   export $var ; 
   echo ${!var}
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
