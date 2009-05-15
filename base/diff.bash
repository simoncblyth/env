# === func-gen- : base/diff.bash fgp base/diff.bash fgn diff
diff-src(){      echo base/diff.bash ; }
diff-source(){   echo ${BASH_SOURCE:-$(env-home)/$(diff-src)} ; }
diff-vi(){       vi $(diff-source) ; }
diff-env(){      elocal- ; }
diff-usage(){
  cat << EOU
     diff-src : $(diff-src)

     diff-dirs dirA dirB

          Render the output of a recursive brief diff 
          more amenable to subsequent processing 
             "diff -r --brief dirA dirB"

        diff /db/hfagc/cdf/cjl/cdf_summer2005_PRL88-071801.xml
        diff /db/hfagc/cdf/cjl/cdf_winter_2006_CDFNOTE7815.xml
        diff /db/hfagc/cdf/cjl/cdf_winter_2006_CDFNOTE7862.xml
        diff /db/hfagc/cdf/cjl/cdf_winter_2006_CDFNOTE7925.xml
        diff /db/hfagc/__contents__.xml
        onlyA /db/hfagc_system
        diff /db/hfagc_tags/__contents__.xml 

        
EOU
}



diff-files-line(){
   local msg="=== $FUNCNAME :"
   [ "$#" != "5" ]      && echo $msg incoorect number of elements && return 1
   [ "$1" != "Files" ]  && echo $msg bad 1st element && return 2
   [ "$3" != "and" ]    && echo $msg bad 3rd element && return 2
   [ "$5" != "differ" ] && echo $msg bad 5th element && return 2
   echo $2:$4
}

diff-only-line(){
   local msg="=== $FUNCNAME :"
   [ "$#" != "4" ]     && echo $msg incoorect number of elements && return 1
   [ "$1" != "Only" ]  && echo $msg bad 1st element && return 2
   [ "$2" != "in" ]    && echo $msg bad 2nd element && return 3
   local d=$3
   [ ${d:${#d}-1} != ":" ] && echo $msg bad last char of 3rd element && return 4
   d=${d:0:${#d}-1}
   echo $d/$4
   # Only in /data/heprez/data/backup/part/cms01.phys.ntu.edu.tw/last/db: hfagc_system
}


diff-dirs(){

   local a=${1}
   local b=${2}

   local cmd="diff -r --brief $a $b "
   $cmd | while read line ; do
       local path
       local paths
       local typ="ERROR"
       case $line in
         Files*) paths=$(diff-files-line $line) ; typ="diff" ;;
          Only*) paths=$(diff-only-line $line)  ; typ="only" ;;
       esac
       if [ "$typ" == "diff" ]; then
           local apath=${paths/:*} ; apath=${apath/$a}
           local bpath=${paths/*:} ; bpath=${bpath/$b}
           [ "$apath" != "$bpath" ] && echo $msg ERROR relative paths do not match && return 1
           path=$apath
       elif [ "$typ" == "only" ]; then
           path=$paths
           [ "${path/$b}" == "$path" ] && typ="${typ}A"
           [ "${path/$a}" == "$path" ] && typ="${typ}B"
           case $typ in
              onlyA) path=${path/$a} ;;
              onlyB) path=${path/$b} ;;
                  *) path="ERROR"    ;;
           esac
       elif [ "$typ" == "ERROR" ]; then
           path=$line
       fi
       echo $typ:$path
   done


}


