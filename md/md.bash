

md-usage(){

cat << EOU

    OSX ONLY METADATA DATABASED SEARCHES as used by  Spotlight

  md-sec/min/hour/day  <-n>   <mdfind-options..>       
  
       List file paths modified in the last n secs/mins/hours/days
       where n defaults to -1 

        md-sec -10              
        md-sec -10 -onlyin $HFAG_SCRAPE_FOLDER/hfagc
    
        md-min       ## all files modified in the last min 
  
        md-min -10                  ## all files modifed in the last 10 mins
        md-min -10 -onlyin .        ##  .. in pwd
  
        md-hour -1                 ## all files modifiedin the last hour
        md-hour -1 -onlyin .       ##  .. in pwd 
    
        md-day -1                 ## all files modified in the last 24 hrs
        md-day -1 -onlyin .        ## .. in pwd   
              
              
  
  md-typ  <filetype> <other args>   
      List file paths with names *.<filetype> 
     eg:
            md-typ tex -onlyin $HOME  
            md-typ xcodeproj
  
  
  md-betwixt
        time range selection
  

EOU



}



md-betwixt(){

  local x1="12:22:00"
  local x2="12:25:00"

  local tt="CreationDate"  
  local day="2008-08-03"
  local tz="+0800"

  local t1="$day $x1 $tz"
  local t2="$day $x2 $tz"
  
  local cmd="mdfind 'kMDItemFS$tt >= \$time.iso(\"$t1\") && kMDItemFS$tt <= \$time.iso(\"$t2\") '"
  echo $cmd
  eval $cmd

}

md-crit(){
  local def=">"
  echo ${MD_CRIT:-$def}
}


md-sec(){
 
  local msg="=== $FUNCNAME :"
  local dsec=-1
  local secs=${1:-$dsec}
  shift

  local cmd="mdfind $* 'kMDItemFSContentChangeDate $(md-crit) \$time.now($secs)' " 

  echo $msg $cmd looking for all files modifed in the last $secs secs \( negative if you want hits \)  > /dev/stderr
  echo $msg arguments after the first are stuffed into mdfind options slot ...  \[-live\] \[-count\] \[-onlyin directory\] > /dev/stderr  
  eval $cmd
}


md-min(){
  local d=-1
  local s=${1:-$d}
  shift
  md-sec $(($s*60)) $*
}

md-hour(){
  local d=-1
  local s=${1:-$d}
  shift
  md-sec $(($s*3600)) $*
}

md-day(){
  local d=-1
  local s=${1:-$d}
  shift
  md-sec $(($s*3600*24)) $*
}




md-typ(){

  local typ=$1
  shift
  local cmd="mdfind $* \"kMDItemFSName == '*.$typ'\" "  

  local
  eval $cmd 
}