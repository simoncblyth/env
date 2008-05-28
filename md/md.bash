

md-usage(){

cat << EOU

  OSX ONLY METADATA DATABASED SEARCHES ... a.k.a : Spotlight

  md-s    <-n>                      : search for files modified in the last n seconds
    examples:
           md-s -600          ## all files modifies in last 10 minutes
           md-s -6000 -onlyin $HFAG_SCRAPE_FOLDER/hfagc
  
  md-typ  <filetype> <other args>   : search for files with names *.<filetype> 
       eg:
            md-typ tex -onlyin $HOME  
  

EOU



}



md-s(){
 
  local msg="=== $FUNCNAME :"
  local dsec=-6000
  local secs=${1:-$dsec}
  shift

  local cmd="mdfind $* 'kMDItemFSContentChangeDate > \$time.now($secs)' " 

  echo $msg $cmd looking for all files modifed in the last $secs secs \( negative if you want hits \) 
  echo $msg arguments after the first are stuffed into mdfind options slot ...  \[-live\] \[-count\] \[-onlyin directory\]  
  eval $cmd

}

md-typ(){

  local typ=$1
  shift
  local cmd="mdfind $* \"kMDItemFSName == '*.$typ'\" "  

  local
  eval $cmd 
}