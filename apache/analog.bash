# === func-gen- : apache/analog fgp apache/analog.bash fgn analog fgh apache
analog-src(){      echo apache/analog.bash ; }
analog-source(){   echo ${BASH_SOURCE:-$(env-home)/$(analog-src)} ; }
analog-srcdir(){   echo $(dirname $(analog-source)) ; }
analog-vi(){       vi $(analog-source) ; }
analog-env(){      elocal- ; }
analog-usage(){
  cat << EOU
     analog-src : $(analog-src)
     analog-dir : $(analog-dir)





     Prepares bite sized chunks of a logfile with corrections applied
     to facilitate np.fromregex parsing, load into python with::

         from env.apache.analog import load
         a = load("/tmp/env/analog/access_log/100000_0")


     analog-chop <numline:100000> <path:~/access_log>

          chop a logfile into chunks of <numline> maximum size 

     analog-chunk <startline:0> <numline:1000> <path:~/access_log>

          writes a chunk from a logfile specified by the startline and 
          number of lines.  Fixes are applied to the chunk to facilitate 
          parsing the chunk into a numpy record array. 
          
     analog-check path

          checks that the python loading of a chunk into a numpy 
          record array succeeds to convert all lines in the chunk  
  

     analog-check-chunks path

          check that all chunks from  original log <path>
          are fully parsable into numpy records



    ISSUES
    ~~~~~~~

       #.  low level truncation difference between chunks and the .chk-original  



EOU
}
analog-dir(){ echo /tmp/env/analog ; }
analog-cd(){  cd $(analog-dir); }
analog-mate(){ mate $(analog-dir) ; }

analog-path(){ echo ${ANALOG_PATH:-~/access_log} ; }
analog-chunk-(){
   local startline=${1:-0}
   local numline=${2:-1000}
   local path=${3:-$(analog-path)}

   # fixes 
   #   1) combine head and tail to make "middle"
   #   2) remove slashes and first colon in apache datetime to allow  numpy.core._mx_datetime_parser to parse 
   #   3) size column is sometimes a dash "-" convert that to 0 to allow size parsing as an integer
   #
   tail -n +${startline} ${path} | head -n ${numline} | perl -p -e 's,(\[\d+)/(\S+)/(\d+):,$1 $2 $3 ,' - | perl -p -e 's,-$,0,' - 
}

analog-chunkdir(){
  local path=${1:-$(analog-path)}
  echo $(analog-dir)/$(basename $path)
}

analog-chunk(){
  local start=${1:-0}
  local numl=${2:-1000}
  local path=${3:-$(analog-path)}
  local name=${numl}_${start}     

  local chunkd=$(analog-chunkdir $path)
  [ ! -d "$chunkd" ] && echo $msg creating $chunkd && mkdir -p $chunkd 
  local chunk=$chunkd/$name 
  echo $msg preparing logfile chunk $chunk 
  analog-chunk- $start $numl > $chunk
}

analog-wc(){  cat ${1:-$(analog-path)} | wc -l ; }

analog-check-chunks(){
   local path=${1:-$(analog-path)}
   local dir=$(analog-chunkdir $path) 
   local chunk
   ls -1 $dir/* | grep -v .chk_orig | while read chunk ; do
       echo $msg $chunk
       #analog-check $chunk
   done
}

analog-check(){ analog-check- $1 $(analog-wc $1) | python ; }
analog-check-(){  cat << EOC
from env.apache.analog import load
a = load( "$1" )
assert len(a) == $2 , ("length mismatch for $1" , len(a), $2 )  
EOC
}


analog-chop(){
  local msg="=== $FUNCNAME :"
  local numl=${1:-100000}
  local path=${2:-$(analog-path)}
  local len=$(analog-wc $path)
  #local len=1705000

  local cmd
  local n
  n=0
  while [ $n -lt $len ]; do
     cmd="analog-chunk $n $numl $path"
     echo $msg $cmd
     eval $cmd
     let n=n+$numl 
  done
}



