aberdeen-vi(){ vi ${BASH_SOURCE:-$(env-home)/aberdeen/aberdeen.bash} ; }
aberdeen-env(){
  local msg="=== $FUNCNAME "
}

roody-(){ . $ENV_HOME/aberdeen/roody.bash && roody-env $* ; }
midas-(){ . $ENV_HOME/aberdeen/midas.bash && midas-env $* ; }
rome-(){  . $ENV_HOME/aberdeen/rome.bash && rome-env $* ; }
abd-(){   . $ENV_HOME/aberdeen/abd.bash && abd-env $* ; }


## NB moving away from this kitchen sink approach ... the envirobnment should bring in 
## only whats needed  

  # roody-
  # rome-
  # abd-
  # midas-
  # 
  # roody-env
  # midas-path > /dev/null
  # roody-path > /dev/null



