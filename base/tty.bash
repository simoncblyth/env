
[ "$BASE_DBG" == "1" ] && echo tty.bash 

## suspect not called
##fix delete key operation in vi
  [ -t 0 ] && stty erase '^?'

## this is the bash equivalent of "bindkey -v"

  if [ "$USER" == "blyth" ]; then
    set -o vi     # vi or emacs CLI editing 
  fi