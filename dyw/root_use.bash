


root-use-conf(){

   ## ftp://root.cern.ch/root/doc/chapter2.pdf
   local rootrc=$HOME/.rootrc
   
   if [ -f "$rootrc" ]; then
     echo === not proceeding, as $rootrc exists ...  first delete $rootrc 
   else
     echo === writing $rootrc
     root-use-rootrc > $rootrc
     cat $rootrc
   fi
}

root-use-rootrc(){

cat << EOC
## do not edit $HOME/.rootrc  here, see source:/trunk/dyw/root_use.bash
Unix.*.Root.MacroPath:      .:$(ROOTSYS)/macros:$HOME/$ENV_BASE/root
EOC

}