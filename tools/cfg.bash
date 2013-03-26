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

Based on http://ajdiaz.wordpress.com/2008/02/09/bash-ini-parser/
works by creating bash functions for each section of an ini file

Generate bash functions with names prefixed by `cfg.` for each section of `~/.my.cnf`::

	[blyth@cms01 tools]$ cfg-parse ~/.my.cnf
	[blyth@cms01 tools]$ cfg-parse ~/.env.cnf

For debugging::

    simon:e blyth$ cfg-;CFGDBG=1 cfg-parse
    [blyth@cms01 ~]$ cfg-;CFGDBG=1 cfg-parse ~/.env.cnf


For accessing a single value use `cfg-get`::

	[blyth@cms01 e]$ cfg-context client ~/.my.cnf
	offline_db_20130315

When need to access multiple values, using `cfg-sect` then multiple `cfg-val` is faster::

	[blyth@cms01 e]$ cfg-sect client
	[blyth@cms01 e]$ cfg-val database
	offline_db_20130315
	[blyth@cms01 e]$ cfg-val password
	censored

	[blyth@cms01 e]$ cfg-sect offline_db   # jump to different section
	[blyth@cms01 e]$ cfg-val database
	offline_db
	[blyth@cms01 e]$ cfg-val host
	dybdb2.ihep.ac.cn

For how to use this from another script/function see the demo functions. 

Internals of how this works::

	[blyth@cms01 tools]$ cfg-parse ~/.my.cnf
	[blyth@cms01 tools]$ cfg.client           # run the generated function, spilling the beans
	[blyth@cms01 tools]$ type cfg.client
	cfg.client is a function
	cfg.client () 
	{ 
	    host=(cms01.phys.ntu.edu.tw);
	    database=(offline_db_20130315);
	    user=(***);
	    password=(***)
	}
	[blyth@cms01 tools]$ echo $database
	offline_db_20130315


TODO
------

#. avoid spilling the beans quite to much


Issue with cms01, the initial read mangles some lines

	[blyth@cms01 ~]$ IFS=$'\n'  && ini=( $(<$path) ) ; for line in ${ini[*]} ; do echo $line ; done
	[fossil]
	repodir = /data/var/scm/fossil
	port = 591
	user = root
	e
	user = blyth
	password = whatever


cms01 is messing up the linearraytest in flaky manner
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


EOU
}



cfg-linearraytest-(){
   local path=$1
   #local ini

   # method1 which is flaky on cms01
   # local IFS=$'\n' 
   # ini=( $(<$path) )

   # method2
   local IFS
   local line
   ini=() ; IFS='' ; while read line ; do ini[${#ini[@]}]=$(echo "$line") ; done < $path

   for line in ${ini[*]} ; do echo "$line" ; done
}

cfg-linearraytest(){
   local path=$1
   local tmp=/tmp/env/$FUNCNAME/$(basename $path) && mkdir -p $(dirname $tmp)
   cfg-linearraytest- $path > $tmp
   diff $path $tmp
}


cfg-parse(){
    local path=${1:-~/.env.cnf}
    export CFG_PATH=$path
    #echo $msg parsing $path into section functions
    # to understand how this works do the below line by line on commandline dumping with:
    #  for line in ${ini[*]} ; do echo $line; done   

    local pfx="_cfg."
    local IFS

    if [ $(hostname) == "cms01.phys.ntu.edu.tw" ]; then 
        IFS='' 
        ini=()  
        while read line ; do ini[${#ini[*]}]=$(echo "$line") ; done < $path
    else
         # this is flaky on cms01
         IFS=$'\n' 
         ini=( $(<$path) )        # convert to line-array
    fi 

    [ -n "$CFGDBG" ] && echo read $path into line array  

    ini=( ${ini[*]//\#*/} )                  # remove comments
    ini=( ${ini[*]/\ =\ /=} )                # remove anything with a space around ' = '
    ini=( ${ini[*]// /} )                    # kill all spaces ... means no meaningful spaves in config values 
    ini=( ${ini[*]//\%\(/\$} )               # replace the %( of python interpolation var %(var)s 
    ini=( ${ini[*]//\)s/} )                  # replace the )s of python interpolation var %(var)s 
    ini=( ${ini[*]/#[/\}$'\n'$pfx} )         # set section prefix by replacing [sectname] with  }\n_cfg.sectname]
    ini=( ${ini[*]/%]/ \(} )                 # convert text2function (1)
    ini=( ${ini[*]/=/=\( } )                 # convert item to array
    ini=( ${ini[*]/%/ \)} )                  # close array parenthesis
    ini=( ${ini[*]/%\( \)/\(\) \{} )         # convert text2function (2)
    ini=( ${ini[*]/%\} \)/\}} )              # remove extra parenthesis

    [ -n "$CFGDBG" ] && echo completed replacements

    ini[0]=''                                # remove first element
    ini[${#ini[*]} + 1]='}'                  # add the last brace

    [ -n "$CFGDBG" ] && echo completed head and tail


    [ -n "$CFGDBG" ] && echo "${ini[*]}"     # echo the generated functions
    eval "$(echo "${ini[*]}")"               # eval the result

    
    # below succeeds to prefix with local, but then have no access duh 
    # hmm maybe
    #    a) generate argument handling allowing controlled disclosure  
    #
    # fin=()
    # local N=${#ini[@]}
    # for ((i=1;i<=$N;++i)) ; do 
    #     local line=${ini[$i]} 
    #     local loc=$(cfg-localize $line) 
    #     fin[$i]=$loc
    # done
    # eval "$(echo "${fin[*]}")"               # eval the localized result
}

cfg-localize(){
    case $1 in
       [^a-zA-Z0-9-]* ) echo $1 ;; 
                    * ) echo local $1 ;;
    esac
}


cfg-vals(){ 
  local keys=$*
  local key
  for key in $keys ; do 
    echo $key $(cfg-val $key)
  done
}
cfg-val(){ echo $(eval "echo \$${1}") ; }              # dynamic access to variable by name
cfg-sect(){ _cfg.${1:-client} > /dev/null 2>&1  ;  }   # dynamic invokation of generated function
cfg-get(){                                             # defines the sect variables, then echos ones of them : when need multiple vals this is inefficient
   cfg-sect $1
   cfg-val $2
}


cfg-context(){
  # node+userid specific context from ini file 
  local sect=${1:-fossil}
  local path=${2}    
  cfg-parse $path
  cfg-sect $sect   
}

cfg-filltmpl-(){
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





cfg-demo(){
   local sect=${1:-client}
   shift
   local keys=$*
   [ -z "$keys" ] && keys="database host user"

   cfg-parse ~/.my.cnf 
   local rc 
   local key
   cfg-sect $sect
   rc=$?
   echo $msg cfg-sect for section \"$sect\" rc $rc

   if [ $rc -eq 0 ]; then
       cfg-vals $keys 
   else
       echo $msg config file $CFG_PATH has no section \"$sect\"  
   fi 
}

cfg-demo2(){
   local sect=${1:-client}
   shift
   [ ! $(cfg-sect $sect) ] &&  cfg-vals $* || echo no sect $sect
}


