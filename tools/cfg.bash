cfg-src(){      echo tools/cfg.bash ; }
cfg-source(){   echo ${BASH_SOURCE:-$(env-home)/$(cfg-src)} ; }
cfg-vi(){       vi $(cfg-source) ; }
cfg-env(){      elocal- ; }
cfg-usage(){ cat << EOU

BASH INI FILE PARSING
========================

Objective:

  * allow bash functions/scripts to access ini files so same config 
    approach can be used for bash/python/C/C++

EOU
}


cfg-context(){
  # node+userid specific context from ini file 
  local sect=${1:-fossil}
  local path=${2:-~/.env.cnf}
  eval $($(env-home)/tools/cnf.py -s $sect -c $path) 
  _cfg_$sect 
  [ -n "$CFGDBG" ] && type _cfg_$sect

}

cfg-filltmpl-(){
  # if filling templates is all you want to do, better to do all in python

  local tmpl=${1:-path-to-template}
  local sect=${2:-cfg-sect-name}
  local path=${3}

  cfg-context $sect $path   # globally polluting context, non-trivial to impinge "local" in func generation

  local IFS=''   # preserve whitespace 
  local line
  while read line ; do
      eval "echo \"$line\" "
  done < $tmpl 
}



