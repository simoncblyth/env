

traccomp-usage(){
   cat << EOU
   
     NB this is just for the "manual" components ... the components that 
     are closely related to repository paths are managed via the autocomponent
     machinery that uses svn "owner" properties in the repository 
   
     
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


traccomp-remove(){
   for c in $* ; do
      SUDO=sudo trac-admin- component remove $c
   done
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





