# === func-gen- : python/tests/time fgp python/tests/time.bash fgn time fgh python/tests
time-src(){      echo python/tests/time.bash ; }
time-source(){   echo ${BASH_SOURCE:-$(env-home)/$(time-src)} ; }
time-vi(){       vi $(time-source) ; }
time-env(){      elocal- ; }
time-usage(){
  cat << EOU
     time-src : $(time-src)
     time-dir : $(time-dir)


EOU
}
time-dir(){ echo $(local-base)/env/python/tests/python/tests-time ; }
time-cd(){  cd $(time-dir); }
time-mate(){ mate $(time-dir) ; }
time-tzs(){
   #echo UTC Asia/Taipei US/Eastern
   local iwd=$PWD
   cd /usr/share/zoneinfo
   local tz
   find . -type f | grep -v .tab | while read tz ; do
     echo ${tz:2}
   done
   cd $iwd
}

time-test-(){ cat << EOT
import time

f = '%Y-%m-%dT%H:%M:%S'
s = 1.278e9
t = time.gmtime( s )                    # construct UTC time from epoch seconds
d = time.strftime( f , t )              # format ... without tz/dst info
p = time.strptime( d , f )              # parse the formatted ... loosing tz/dst info
c = time.mktime( p ) - time.timezone

print ( time.tzname, time.timezone, time.daylight , s , t, d, p, c ) 
assert s == c 

EOT
}

time-test(){
   for tz in $(time-tzs) ; do
      $FUNCNAME- | TZ=$tz python || echo $msg ERROR for $tz 
   done
}

