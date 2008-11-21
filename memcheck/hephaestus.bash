hephaestus-env(){
  echo -n 
}

hephaestus-usage(){
  cat << EOU
     hephaestus-get   :   trivial approach to downloading a few levels of viewcvs 

EOU

}

hephaestus-url(){   echo http://atlas-sw.cern.ch/cgi-bin/viewcvs-atlas.cgi/offline/Control/$1/$2?view=co ; }
hephaestus-bases(){ echo Hephaestus Hephaestus/cmt  ; }

hephaestus-files(){
   local base=$1
   local sub=$2
   if [ "$base" == "Hephaestus" ]; then
      case $sub in
              DIRS) echo src python cmt Hephaestus ;;
               src) echo Utils.c MemoryTracker.c MemoryTrace.c HashTable.c GenericBacktrace.c DebugInfo.c ;;
            python) echo __init__.py  ;;
               cmt) echo  requirements ;;
        Hephaestus) echo  Utils.h MemoryTrace.h Hephaestus.h HashTable.h GenericBacktrace.h DebugInfo.h CheckPoints.h ;;
      esac
   elif [ "$base" == "Hephaestus/cmt" ]; then
       case $sub in 
            DIRS) echo fragments ;;
       fragments) echo python_extension_header python_extension ;;
       esac
   fi    
}

hephaestus-get(){
  for base in $(hephaestus-bases) ; do
    for dir in $(hephaestus-files $base DIRS) ; do
       for file in $(hephaestus-files $base $dir) ; do
          local url=$(hephaestus-url $base/$dir $file)
          local path=$base/$dir/$file
          echo url $url path $path 
          mkdir -p $(dirname $path)
          curl -o $path $url
       done
    done
  done
}




