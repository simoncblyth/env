
tracperm-source(){ echo $BASH_SOURCE ; }
tracperm-vi(){     vi $(tracperm-source) ; }
tracperm-usage(){

  cat << EOU
  
      These functions are usually used with prefix SUDO=sudo 
  
        TRAC_INSTANCE : $TRAC_INSTANCE
        tracperm-level : $(tracperm-level) 
  
      NB despite first appearances, to target a non-default instance 
      you must use TRAC_INSTANCE=whatever prefix , due to usage
      of trac-admin- under the covers which requires this 
   
       tracperm--  args
            run the args in a "sudo bash -c" sub-environment 
  
       tracperm-prepare-all
             sets the permissions for all the instances    
                 tracperm-- tracperm-prepare-all
                 
             former approach :    
                 SUDO=sudo tracperm-prepare-all
  
       tracperm-prepare 
             sets permissions for a single instance, namely TRAC_INSTANCE
             target non default instance with eg:
                 tracperm-- TRAC_INSTANCE=newtest tracperm-prepare
                 
             former approach giving "sudo python" troubles ... see #111    
                 SUDO=sudo TRAC_INSTANCE=aberdeen tracperm-prepare
                 
  
       tracperm-level <name defaults to TRAC_INSTANCE> 
             security level for the named instance
  
       tracperm-set <anonymous|authenticated|blyth|...>  WHATEVER_PERMISSION OTHER_ETC
             set permissions for a user by first removing all permissions and then adding
             the named ones
             
    
    
EOU

}

tracperm-env(){
  echo -n
}

tracperm--(){
   sudo bash -c "export ENV_HOME=$ENV_HOME ; . $ENV_HOME/env.bash ; env- ; trac- ; tracperm- ; $* "
}

tracperm-set(){

  local user=${1:-anonymous}
  shift 

  echo $msg TRAC_INSTANCE:[$TRAC_INSTANCE] SUDO:[$SUDO] user:[$user] perms [$*]  

  trac-admin- permission remove $user \'*\'
  trac-admin- permission add    $user $* 
}


tracperm-level(){
    case ${1:-$TRAC_INSTANCE} in 
                               env|tracdev) echo loose ;;
       heprez|aberdeen|dybsvn|data|newtest) echo tight ;;   
                                  workflow) echo paranoid ;;
                                         *) echo paranoid ;;
    esac
}


tracperm-prepare(){

    local msg="=== $FUNCNAME :"
	local level=$(tracperm-level $TRAC_INSTANCE)

    echo $msg setting perms for TRAC_INSTANCE:[$TRAC_INSTANCE] to level:[$level] SUDO:[$SUDO]

    local views="WIKI_VIEW TICKET_VIEW BROWSER_VIEW LOG_VIEW FILE_VIEW CHANGESET_VIEW MILESTONE_VIEW ROADMAP_VIEW REPORT_VIEW"	 
    local other="TIMELINE_VIEW SEARCH_VIEW"
	
    local hmmm="CONFIG_VIEW"
    local wiki="WIKI_CREATE WIKI_MODIFY"
    local ticket="TICKET_CREATE TICKET_APPEND TICKET_CHGPROP TICKET_MODIFY"
    local milestone="MILESTONE_CREATE MILESTONE_MODIFY"
    local report="REPORT_SQL_VIEW REPORT_CREATE REPORT_MODIFY"
    local fullblog=""
    case $NODE_TAG in 
      N)  fullblog="BLOG_VIEW BLOG_CREATE BLOG_MODIFY_OWN BLOG_COMMENT" ;;
    esac

    ## remove WIKI_DELETE MILESTONE_DELETE REPORT_DELETE ... leave those to admin only
    ## allow unauth to REPORT_VIEW 
 

    if [ "$level" == "loose" ]; then
 
      tracperm-set anonymous     "$views $other" 
      tracperm-set authenticated "$views $other $hmmm $wiki $ticket $milestone $report $fullblog"
      
    elif [ "$level" == "tight" ]; then
    
      # anonymous user can only WIKI_VIEW ... but make sure they can they login ?
	
      tracperm-set anonymous     "WIKI_VIEW"
      tracperm-set authenticated "$views $other $hmmm $wiki $ticket $milestone $report $fullblog "
  
    elif [ "$level" == "paranoid" ]; then	  
		      
      tracperm-set anonymous     "WIKI_VIEW"
      tracperm-set authenticated "$views $other $hmmm $wiki $ticket $milestone $report $fullblog "
		
    else
        echo "ERROR security level $level is not implemented "
    fi            
     
    tracperm-set $(trac-administrator) TRAC_ADMIN   ## TRAC_ADMIN means all permisssions 
        
    case $TRAC_INSTANCE in        
       env|dybsvn|workflow) tracperm-bitten ;;
                         *)   echo -n       ;;
    esac
    
    trac-admin- permission list 
    
}


tracperm-bitten(){

 
  trac-configure components:bitten.\*:enabled

 

  trac-admin- permission add $(trac-administrator) BUILD_ADMIN
  trac-admin- permission add authenticated BUILD_VIEW
  trac-admin- permission add authenticated BUILD_EXEC

}




tracperm-prepare-all(){

   for name in $(trac-instances)
   do
      TRAC_INSTANCE=$name tracperm-prepare
   done

}

