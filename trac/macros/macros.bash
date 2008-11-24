
tmacros-usage(){

cat << EOU

     http://trac.edgewall.org/wiki/WikiMacros

     TRAC_MACROS_DIR : $TRAC_MACROS_DIR
     APACHE2_USER    : $APACHE2_USER

     macros-place  <name>    :   
          propagate all updated wiki macros from \$TRAC_MACROS_DIR into the plugins 
          folder of the named tracitory


EOU




}


tmacros-env(){

  export TRAC_MACROS_DIR=$ENV_HOME/trac/macros
  elocal-
}


tmacros-status(){

   local name=$1
   local plugins=$(tmacros-plugins $name)
   local py=$2
   
   [ ! -f $plugins/$py ]    && return 1
   [ $py -nt $plugins/$py ] && return 2
   return 0
}


tmacros-plugins(){
   local name=$1
   echo $SCM_FOLD/tracs/$name/plugins
}


tmacros-place(){

  local msg="=== $FUNCNAME :"
  local name=$1
  local plugins=$(tmacros-plugins $name)
  local iwd=$PWD
  
  local dir=$ENV_HOME/trac/macros
  cd $dir
  for py in *.py
  do
      local copy=0
      tmacros-status $name $py
      case $? in
         0) echo $msg $py is uptodate ;;
         1) echo $msg first copy of $py && copy=1 ;;
         2) echo $msg update $py && copy=1 ;;   
         *) echo $msg ERROR $? && return ;;
       esac  
       
       if [ "$copy" == "1" ]; then
          local cmd="sudo -u $APACHE2_USER cp $py $plugins/$py "
          echo $cmd
          eval $cmd
      fi 
      
  done

  cd $iwd
}


