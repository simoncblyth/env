# === func-gen- : base/terminal/terminal fgp base/terminal/terminal.bash fgn terminal fgh base/terminal
terminal-src(){      echo base/terminal/terminal.bash ; }
terminal-source(){   echo ${BASH_SOURCE:-$(env-home)/$(terminal-src)} ; }
terminal-vi(){       vi $(terminal-source) ; }
terminal-env(){      elocal- ; }
terminal-usage(){ cat << EOU

$FUNCNAME
=================

Use function keys as standins for failing keys 
------------------------------------------------

1. Terminal > Preferences... [Keyboard] 
2. Click on F1/F2/... enter keys using the virtual keyboard



EOU
}




