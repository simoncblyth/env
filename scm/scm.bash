
scm-usage(){
  
   cat << EOU

      An SCM is a combination of an SVN repository and a Tracitory , this provides
      commands to create these two tied objects in a coherent way.


      scm-create  <name> <arg> 
               
           Create a repository+tracitory (an scm) named <name> and if <arg> is 
           a valid directory path then import the content into it
           if <arg> is INIT then just create the branches/tags/trunk or if EMPTY
           the default leave the repository at revision 0

           For example : 
                scm-create data INIT         ## with branches/tags/trunk at revision 1
                scm-create data EMPTY        ## at revision 0 
                scm-create data /path/to/directory/to/put/into/trunk
                

      scm-wipe <name>
      
           Delete the repository + trac instance called <name>
                  
                  
                  
                        
                                    
        NOTES ...
           * transition from old to new is as yet incomplete          
           * old way is overly complicated by attempting to support remote creation ...
        
          OLD STRUCTURE
          ================                                      
                      
              the initenv is done in   
                   oscm/scm-use.bash::scm-use-create-local
              which is called from 
                    oscm/scm.bash::scm-create
   
   
            
EOU

}

scm-env(){
  elocal-
}


scm-create(){

  local msg="=== $FUNCNAME :" 
  local name=$1
  shift 
  [ -z "$name" ]     && echo $msg an instance name must be provided && return 1

  svn-
  svn-create $name $*
  
  trac-
  trac-create $name 
   

}


scm-wipe(){
  local msg="=== $FUNCNAME :" 
  local name=$1
  shift 
  [ -z "$name" ]     && echo $msg an instance name must be provided && return 1


  svn-
  svn-wipe $name
  
  trac-
  trac-wipe $name    


}


