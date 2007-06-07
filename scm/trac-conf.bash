


trac-conf-perm(){
   local name=${1:-$SCM_TRAC}
   shift
   echo $SUDO trac-admin $SCM_FOLD/tracs/$name permission $*
        $SUDO trac-admin $SCM_FOLD/tracs/$name permission $*
}


trac-conf-set-perms(){
   name=${1:-$SCM_TRAC}
   user=${2:-anonymous}
   shift 
   shift    
   ## remove all permissions first ... and then apply 
   trac-conf-perm $name remove $user  \'*\'
   trac-conf-perm $name add $user $*
}


trac-conf-perms(){

    local name=${1:-$SCM_TRAC}
    local level=${2:-$SCM_SECURITY_LEVEL}

	views="WIKI_VIEW TICKET_VIEW BROWSER_VIEW LOG_VIEW FILE_VIEW CHANGESET_VIEW MILESTONE_VIEW ROADMAP_VIEW REPORT_VIEW"	 
    other="TIMELINE_VIEW SEARCH_VIEW"
	hmmm="CONFIG_VIEW"
    wiki="WIKI_CREATE WIKI_MODIFY"
	ticket="TICKET_CREATE TICKET_APPEND TICKET_CHGPROP TICKET_MODIFY"
    milestone="MILESTONE_CREATE MILESTONE_MODIFY"
    report="REPORT_SQL_VIEW REPORT_CREATE REPORT_MODIFY"
  
    ## remove WIKI_DELETE MILESTONE_DELETE REPORT_DELETE ... leave those to admin only
    ## allow unauth to REPORT_VIEW 
 
    if [ "$level" == "loose" ]; then
 
      trac-conf-set-perms $name anonymous     "$views $other" 
	  trac-conf-set-perms $name authenticated "$views $other $hmmm $wiki $ticket $milestone $report"
      trac-conf-set-perms $name admin TRAC_ADMIN
 
    elif [ "$level" == "tight" ]; then
    
      ## anonymous user can do nothing ... but can they login ? 
      trac-conf-set-perms $name anonymous     "" 
	  trac-conf-set-perms $name authenticated "$views $other $hmmm $wiki $ticket $milestone $report"
      trac-conf-set-perms $name admin TRAC_ADMIN 
      
    else
        echo "ERROR security level $level is no implemented "
    fi            
       
    ## does TRAC_ADMIN include XML_RPC ? yes 
}



trac-conf-component(){ 
   local name=${1:-$SCM_TRAC}
   shift
   echo sudo trac-admin $SCM_FOLD/tracs/$name component $*
        sudo trac-admin $SCM_FOLD/tracs/$name component $*
}

trac-conf-components(){

    local name=${1:-$SCM_TRAC}
    local pairs="red:admin green:admin blue:admin"

    ## remove the default components 
    trac-conf-component $name remove component1
    trac-conf-component $name remove component2

    for pair in $pairs
    do
	    local component=$(echo $pair | cut -f1 -d:)
	    local     owner=$(echo $pair | cut -f2 -d:)
        trac-conf-component $name add $component $owner
    done	   
    trac-conf-component $name list 
}






trac-conf-notification(){
   echo simply make the following settings ... to get emails on ticket changes
   echo the user settings should include the email address ... if doesnt correspond to default domain
cat << EOX
smtp_default_domain = hep1.phys.ntu.edu.tw
smtp_enabled = true
always_notify_owner = true
always_notify_reporter = true
always_notify_updater = true
EOX
   echo sudo vi $SCM_FOLD/tracs/env/conf/trac.ini
}

trac-conf-authz-check(){
  find $SCM_FOLD/tracs -name trac.ini -exec grep -H auth {} \;
}





