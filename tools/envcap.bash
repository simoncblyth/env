# === func-gen- : tools/envcap fgp tools/envcap.bash fgn envcap fgh tools
envcap-src(){      echo tools/envcap.bash ; }
envcap-source(){   echo ${BASH_SOURCE:-$(env-home)/$(envcap-src)} ; }
envcap-vi(){       vi $(envcap-source) ; }
envcap-env(){      elocal- ; }
envcap-notes(){ cat << EON



    env | sort | perl -n -e 'm,([^=]*)=(.*), && print "export $1=$2\n" ' - > /tmp/encap.sh 
    cat ~/env-fast.sh | sort > /tmp/env-fast-sorted.sh
    diff /tmp/encap.sh /tmp/env-fast-sorted.sh


EON
}
envcap-dir(){ echo $(local-base)/env/tools/envcap ; }
envcap-cd(){  cd $(envcap-dir); }
envcap-mate(){ mate $(envcap-dir) ; }

envcap-dir(){ echo /tmp/$USER/envcap ; } 
envcap-bef(){ echo $(envcap-dir)/envcap-bef.sh ; }
envcap-aft(){ echo $(envcap-dir)/envcap-aft.sh ; }
envcap-fst(){ echo $(envcap-dir)/envcap-fst.sh ; }
envcap-usage(){ cat << EOU

   envcap <some command to setup environment>

        capture the environment difference as a result of
        running some setup command

   envcap-dif

        create $(envcap-fst) from the diff of before and after env dumps

   envcap-pathfix

        flawed approach to special handling of PATH variables.
        The flaw is that it assumes pre-fixing and post-fixing to 
        the PATH only. But that is not the case for NuWa CMT.

   envcap-paths

        list names of envvars that contain PATH in key or value


EOU
}

envcap(){
   local dir=$(envcap-dir)
   mkdir -p $dir
   env > $(envcap-bef)
   eval $*
   env > $(envcap-aft)
}
envcap-paths(){ env | grep PATH | perl -n -e 'm,(.*PATH)=, && print "$1\n" ' - ; }

envcap-dif-(){
  diff $(envcap-bef) $(envcap-aft) 
}
envcap-dif(){
  $FUNCNAME- | perl -n -e 'm/^> (.*)$/ && print "export $1\n" ' - | grep -v "PWD="  > $(envcap-fst)
}

envcap-befv(){ perl -n -e "m,^$1=(\S*), && print \$1"  $(envcap-bef) ; }
envcap-pathfix-cmd(){ 
   local var=$1
   local bef=$(envcap-befv $var)
   cat << EOC
perl -pi -e "s,export $var=(\S*)$bef(\S*),export $var=\$1\\\$$var\$2," $(envcap-fst)
EOC
}

envcap-pathfix(){
  local var 
  envcap-paths | while read var ; do
       local bef=$(envcap-befv $var)
       echo $var $bef
       local cmd=$(envcap-pathfix-cmd $var)
       echo $cmd
       eval $cmd
  done
}




