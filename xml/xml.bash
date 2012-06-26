xml-usage(){ cat << EOU

EOU
}

exist-(){   [ -r $ENV_HOME/xml/exist.bash ] && . $ENV_HOME/xml/exist.bash && exist-env $* ; }
modjk-(){   [ -r $ENV_HOME/xml/modjk.bash ] && . $ENV_HOME/xml/modjk.bash && modjk-env $* ; }

