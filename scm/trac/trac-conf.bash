
trac-conf-usage(){

cat << EOU

   trac-conf-perm <name> list/add/remove/...   <perms>     
                     use trac-admin to manipulate permissions for the <name> tracitory 

   trac-conf-set-perms  <name> <username> PERM1 PERM2 ..
                     remove all permissions then add the listed perms
   
   trac-conf-perms  <name> <loose|tight|paranoid>
   
   trac-conf-intertrac 

   trac-conf-component <name> <list|add|..>
                     use trac-admin to manipulate the components 
					 	
   trac-conf-components <name> red:admin green:admin blue:admin


   trac-conf-notification 
                     guided editing to enable email notification on ticket changes

EOU


}



#
#     header_logo:link:http://example.org/
#     header_logo:src:common/trac_banner.png 
#     project:descr:My Example Project
#     project:icon:common/trac.ico
#    
#
#      header_logo:link:$SCM_URL/tracs/env
#


trac-conf-env(){

   elocal-
   trac-
}



trac-conf-perm(){
   local name=${1:-$SCM_TRAC}
   shift
   echo $SUDO trac-admin $SCM_FOLD/tracs/$name permission $*
        $SUDO trac-admin $SCM_FOLD/tracs/$name permission $*
}


trac-conf-set-perms(){
   local name=${1:-$SCM_TRAC}
   local user=${2:-anonymous}
   shift 
   shift    
   ## remove all permissions first ... and then apply 
   trac-conf-perm $name remove $user  \'*\'
   trac-conf-perm $name add $user $*
}


trac-conf-perms(){

    local name=${1:-$SCM_TRAC}
    local level=${2:-$SCM_SECURITY_LEVEL}

	local views="WIKI_VIEW TICKET_VIEW BROWSER_VIEW LOG_VIEW FILE_VIEW CHANGESET_VIEW MILESTONE_VIEW ROADMAP_VIEW REPORT_VIEW"	 
    local other="TIMELINE_VIEW SEARCH_VIEW"
	local hmmm="CONFIG_VIEW"
    local wiki="WIKI_CREATE WIKI_MODIFY"
	local ticket="TICKET_CREATE TICKET_APPEND TICKET_CHGPROP TICKET_MODIFY"
    local milestone="MILESTONE_CREATE MILESTONE_MODIFY"
    local report="REPORT_SQL_VIEW REPORT_CREATE REPORT_MODIFY"
  
    ## remove WIKI_DELETE MILESTONE_DELETE REPORT_DELETE ... leave those to admin only
    ## allow unauth to REPORT_VIEW 
 
    if [ "$level" == "loose" ]; then
 
      trac-conf-set-perms $name anonymous     "$views $other" 
	  trac-conf-set-perms $name authenticated "$views $other $hmmm $wiki $ticket $milestone $report"
      trac-conf-set-perms $name admin TRAC_ADMIN
 
    elif [ "$level" == "tight" ]; then
    
      ## anonymous user can do nothing ... but can they login ?
      ## use the restricted area access workaround to avoid the error on arrival issue
      ## 
      trac-conf-set-perms $name anonymous     "WIKI_VIEW"
	  ##trac-conf-set-perms $name authenticated "RESTRICTED_AREA_ACCESS $views $other $hmmm $wiki $ticket $milestone $report"
      ## I think RESTRICTED_AREA_ACCESS is not being use, but is the workaround above alluding to smth I have forgotten ?
      trac-conf-set-perms $name authenticated "$views $other $hmmm $wiki $ticket $milestone $report"
      trac-conf-set-perms $name admin TRAC_ADMIN 
  
    elif [ "$level" == "paranoid" ]; then	  
		      
	  trac-conf-set-perms $name anonymous     "WIKI_VIEW"
	  ##trac-conf-set-perms $name authenticated "RESTRICTED_AREA_ACCESS $views $other $hmmm $wiki $ticket $milestone $report"
      ## I think RESTRICTED_AREA_ACCESS is not being use, but is the workaround above alluding to smth I have forgotten ?
      trac-conf-set-perms $name authenticated "$views $other $hmmm $wiki $ticket $milestone $report"
	  trac-conf-set-perms $name blyth TRAC_ADMIN
	  
			  
			  
    else
        echo "ERROR security level $level is no implemented "
    fi            
     
    trac-conf-perm $name list    
           
    ## does TRAC_ADMIN include XML_RPC ? yes 
}



trac-conf-intertrac(){

    local self=${1:-$SCM_TRAC}    
    for path in $SCM_FOLD/tracs/*
    do
       local target=$(basename $path)
       if ([ -d "$path" ] && [ "X$target" != "X$self" ]) then
          local abbr=$(echo $target | perl -pe 's/_release//')
          local conf
          if [ "X$abbr" == "X$target" ]; then
             ## not a release , so do not abbreviate
             conf="intertrac:$target.title:$target intertrac:$target.url:/tracs/$target intertrac:$target.compat:false"
          else   
             conf="intertrac:$abbr:$target intertrac:$target.title:$target intertrac:$target.url:/tracs/$target intertrac:$target.compat:false"
          fi  
          echo $target $conf
          ini-edit $SCM_FOLD/tracs/$self/conf/trac.ini $conf 
       else
          echo skip self $target
       fi  
    done

}



trac-conf-component(){ 
   local name=${1:-$SCM_TRAC}
   shift
   echo sudo trac-admin $SCM_FOLD/tracs/$name component $*
        sudo trac-admin $SCM_FOLD/tracs/$name component $*
}

trac-conf-components(){

 
    
    local name=${1:-$SCM_TRAC}
    shift 
    
    local pairs=$*
   
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





