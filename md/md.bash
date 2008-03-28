
md-s(){
 
  local msg="=== $FUNCNAME :"
  local dsec=-6000
  local secs=${1:-$dsec}
  shift

  local cmd="mdfind $* 'kMDItemFSContentChangeDate > \$time.now($secs)' " 

  #  osx only Spotlight based search, using metadata database in tiger and leopard
  # usage example :
  #    md-s -6000 -onlyin $HFAG_SCRAPE_FOLDER/hfagc
  # 

  echo $msg $cmd looking for all files modifed in the last $secs secs \( negative if you want hits \) 
  echo $msg arguments after the first are stuffed into mdfind options slot ...  \[-live\] \[-count\] \[-onlyin directory\]  
  eval $cmd

}

