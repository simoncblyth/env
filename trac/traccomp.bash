

traccomp-usage(){
   cat << EOU
   
      traccomp-add <path>   defaulting to $(traccomp-path)
      
            add components/owners to trac instance using trac-admin- 
            reading from a file with format 
            
               a title possibly with spaces : blyth      
               secondtitle : offline 
   
      traccomp-clear 
            remove the trac default components
   
      traccomp-prepare
             -clear and -add the default components  
   
   
EOU

}

traccomp-env(){
   trac-
}

traccomp-prepare(){
   traccomp-clear
   traccomp-add
}


traccomp-clear(){
   SUDO=sudo trac-admin- component remove component1
   SUDO=sudo trac-admin- component remove component2
}

traccomp-path(){
   echo $ENV_HOME/trac/nuwacomp.txt
}


traccomp-default-owner(){
   echo offline
}


traccomp-add(){

   local msg="=== $FUNCNAME :" 
   local path=${1:-$(traccomp-path)}
   
   echo $msg reading from $path   
   cat $path | while read line   
   do
        local field=0
        local name=""
        local owner=""
       
        
        for wd in $line ; do 
           if [ "$wd" == ":" ]; then
              field=$(($field + 1))
           else
              case $field in 
                 0) [ -z "$name" ] && name="$wd" || name="$name $wd" ;;
                 1) owner="$owner$wd" ;; 
                 *) left="$left$wd" ;;
              esac   
           fi 
        done
        [ -z "$owner" ] && owner=$(traccomp-default-owner)
        
         echo $msg $line ===\> owner [$owner] name [$name]
        
        [ -z "$name" -o "$name" == " " ] ||  \
              $SUDO trac-admin $(TRAC_INSTANCE=dybsvn trac-envpath) component add  "$name" $owner
        
        ## trac-admin- and arguments with spaces cause problems 
   done 

}


traccomp-owner(){
   local owner=$(svn propget owner $1 2>/dev/null) 
   [ -n "$owner" ] && owner=": $owner"
   echo $owner
}

traccomp-from-wc(){

    local path=$1
    local proj=$(basename $path)
    
    echo 
    echo $proj / $(traccomp-owner $path) 
    [ -n "$TRACCOMP_BRIEF" ] && return 0
    
    local iwd=$PWD
    cd $path
    for name in $(ls -1) ; do
           case $name in 
     cmt|InstallArea)  echo -n  ;;
                   *) [ -d $name ]  && echo $proj / $name $(traccomp-owner $name)   ;; 
            
           esac
    done
    cd $iwd

}



