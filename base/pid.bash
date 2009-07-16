# === func-gen- : base/pid fgp base/pid.bash fgn pid
pid-src(){      echo base/pid.bash ; }
pid-source(){   echo ${BASH_SOURCE:-$(env-home)/$(pid-src)} ; }
pid-vi(){       vi $(pid-source) ; }
pid-env(){      elocal- ; }
pid-usage(){
  cat << EOU
     pid-src : $(pid-src)


     pid-ps 
          piping to cat avoids truncation to terminal width 



     Playing with the shell ...
          http://www.farside.org.uk/200805/stupid_unix_tricks

      echo $((1 << 64))
      1

      echo $((0xFF))
      255



      Return to prior dir :          

    simon:030307 blyth$ cd -
/var/scm/backup/dayabay/tracs


      Expand prior dir :

    simon:tracs blyth$ echo ~-
/var/scm/backup/dayabay/tracs/dybsvn/2008/11/10/030307



EOU
}


pid-path(){ echo /proc/$1/status ; }

pid-vmsize-(){
   local pid=$1
   local path=$(pid-path $pid)
   while true
   do
       [ -f $path ] && grep VmSize $path || return 0
       sleep 5
   done
}

pid-tree(){ pstree -Gpl ; }
pid-ps(){ ps -eauxf | cat ; }
pid--(){ pid-ps ; }

