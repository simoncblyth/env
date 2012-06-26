
cfg-usage(){ cat << EOU
#
#   http://ajdiaz.wordpress.com/2008/02/09/bash-ini-parser/
#
#     to debug this script and the  .ini file it converts into bash functions
#     use :
#          CFGDBG=1 . cfg.bash
#
#
EOU
}


cfg.parser () {
    local IFS=$'\n' && ini=( $(<$1) )        # convert to line-array
    ini=( ${ini[*]//\#*/} )                  # remove comments
    ini=( ${ini[*]/\ =\ /=} )                # remove anything with a space around ' = '
    ini=( ${ini[*]// /} )                    # kill all spaces ... means no meaningful spaves in config values 
    ini=( ${ini[*]//\%\(/\$} )               # replace the %( of python interpolation var %(var)s 
    ini=( ${ini[*]//\)s/} )                  # replace the )s of python interpolation var %(var)s 
    ini=( ${ini[*]/#[/\}$'\n'cfg.section.} ) # set section prefix
    ini=( ${ini[*]/%]/ \(} )                 # convert text2function (1)
    ini=( ${ini[*]/=/=\( } )                 # convert item to array
    ini=( ${ini[*]/%/ \)} )                  # close array parenthesis
    ini=( ${ini[*]/%\( \)/\(\) \{} )         # convert text2function (2)
    ini=( ${ini[*]/%\} \)/\}} )              # remove extra parenthesis
    ini[0]=''                                # remove first element
    ini[${#ini[*]} + 1]='}'                  # add the last brace
    [ -n "$CFGDBG" ] && echo "${ini[*]}"     # echo the generated functions
    eval "$(echo "${ini[*]}")"               # eval the result
}

## generate and evaluate bash functions for each section in the ini 
cfg.parser ~/.dybdb.ini

## invoke a section function to define the variables $var1 etc..
cfg.section.testdb




