open-source(){   echo ${BASH_SOURCE} ; }
open-vi(){       vi $(open-source) ; }
open-env(){      elocal- ; }
open-usage(){ cat << EOU

open-page N /path/to/book.pdf
    open specific page of a pdf using Preview.app, using simple applescript 


EOU
}


open-page-notes(){ cat << EON

https://apple.stackexchange.com/questions/233945/opening-a-specific-page-on-mac-preview-from-terminal

1. Need to enable access in system prefs on first use

2. Usually there is an offset between PDF "file" and "print" pages, so 
   create bash functions for frequently referred to books 
   to translate from the printed page numbers to the file pages.::

        mybook(){   open- ; open-page $(( $1 + 20 )) /path/to/mybook.pdf  ; }

EON
}

open-page-(){ cat << EOP

tell application "Preview" to activate
delay 0.25

tell application "System Events" to tell process "Preview" to click menu item "Go to Page…" of menu "Go" of menu bar 1
delay 0.25
        
tell application "System Events" to keystroke "${1:-0}" 
delay 0.25

tell application "System Events" to key code 36

EOP
}

open-page(){

   local page=${1:-1}
   local pdf=$2

   [ ! -f "$pdf" ] && echo $msg no pdf $pdf && return 

   local tmp=/tmp/$USER/env/osx/${FUNCNAME}-${page}.applescript
   [ ! -d $(dirname $tmp) ] && mkdir -p $(dirname $tmp)
   [ ! -f $tmp ] && $FUNCNAME- $page > $tmp

    open -a Preview "$pdf"
    sleep .5

    osascript $tmp 
}



