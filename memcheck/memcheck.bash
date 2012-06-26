memcheck-usage(){ cat << EOU

EOU
}
memcheck-env(){ echo -n ; }

hephaestus-(){ . $ENV_HOME/memcheck/hephaestus.bash && hephaestus-env $* ; }
