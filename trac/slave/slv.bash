# === func-gen- : base/slv fgp base/slv.bash fgn slv fgh base
slv-src(){      echo base/slv.bash ; }
slv-source(){   echo ${BASH_SOURCE:-$(env-home)/$(slv-src)} ; }
slv-vi(){       vi $(slv-source) $(slv-slave-path) $(slv-runtest-path) $(slv-isotest-path) ; }

slv-srcdir(){
   ## dev versions ... 
   #echo $(env-home)/offline
   ##  operational ones
   echo $DYB/installation/trunk/dybinst/scripts
}


slv-isotest-path(){ echo $(slv-srcdir)/isotest.sh ; }
slv-runtest-path(){ echo $(slv-srcdir)/runtest.sh ; }
slv-slave-path(){   echo $(slv-srcdir)/slave.sh ; }
slv-recipe-path(){  echo $(slv-srcdir)/${1:-dybinst}.xml ; }

slv-usage(){
  cat << EOU

   == GOALS ==

        simplify the slave- funcs ...  with an eye to doing daily builds under slave control
           a) work in fully relative manner ... 
           b) minimize config 
           c) do it without the dyb__ funcs ... that are single install fixated
           d) environment isolation ... do it in env -i for insensitivity to calling environment
       
           e) * .cfg must not be web-accessible so cannot live in the build dir 

           f) unify builds : daily/weekly/update to use single bitten-slave machinery/reporting 
              with minor differences 
                    - build invokation directory 
                          * update build into fixed dir 
                          * weekly? green field build into empty dir
                    - external option 
                          * daily? build using externals shared with the update build


   == HOW TO DO A LOCAL TEST ==


    Switch to local and restart the slave ...
        vi ~/.dybinstrc   ## change buildsurl to local        
        sv restart dybslv 
        sv > tail -f dybslv

   Switch back ...

        vi ~/.dybinstrc   ## builds url back to master trac
        sv restart dybslv



   == OBSERVATIONS ==

     While the master is down the slave dies with :
         <urlopen error (104, 'Connection reset by peer')>


   == fixed interval bitten-slave operation ==

      IDEA A : 
         * use single shot bitten-slave invokation
         * make the requests to the master at the "right" time based on timeline observations 
              * calling at fixed times would not work if coincides with commit cooldown windows
         * observe the timeline rss feed to time the request
              * http://www.feedparser.org/
 
            ==> this requires duplication of the master slave sheduling 

     IDEA B :

         * at the allotted time ... start asking the master if there is a build pending 
          ... up until the send of a time window for doing this type of build
 
            ==>  could implement by adding an "askmax" option to bitten-slave  
   
   == on logurl ==

      formerly primed env with 
         NUWA_LOGURL BUILD_NUMBER BUILD_CONFIG
      which is used by dybinst to construct the url 
       ...   but that wont be appropiate in all cases 
           as the layout will be different for flavors of slave
 
       better to construct this inside the recipe


   
           nginx-ln $DYB/logs
           nginx-ln $DYB/NuWa-trunk/dybgaudi/Documentation/OfflineUserManual/tex manual

                                    
   == TODO : ==

       1) propagate NUWA_LOGURL via dybinst local config rather than environment 
          to keep this out of the recipe and config

       3) placement of outputs/logs ... caution regards credentials and web access

       4) migrate the update "standard" slave run to use recipes generated here  
         
              * system python needs bitten installed 
              * perhaps with patched extra option from me
          
          downside
              * slave cannot then start from an empty dir  
                    (modulo a dybinst config file providing credentials for the slave )


   == DEV FUNCS ==
    
     slv-recipe-update <dybinst|local.dybinst>
           update slv-recipe-path : $(slv-recipe-path)


     slv-recipe
           emit build/test recipe to stdout 
           normally the recipe is kept on the master ... but convenient to keep all together for dev  
           for svn:export of a file use "dir" attribute for the name of the file 
           current bitten on cms01 doesnt recognize username/password attributes ... so only working due to svn auth cache 

   == notes ... ==

          pstree -al $(pgrep -n screen)


EOU
}
slv-env(){      
   elocal-  
   private-
}
slv-cd(){  cd $(slv-dir); }

slv-name(){ hostname -s ; }
slv-repo(){   private-val SLV_REPO ; }
slv-builds(){ private-val $(echo SLV_$(slv-repo)_BUILDS | private-upper ) ; }
slv-user(){   private-val $(echo SLV_$(slv-repo)_USER   | private-upper ) ; }
slv-pass(){   private-val $(echo SLV_$(slv-repo)_PASS   | private-upper ) ; }
slv-url(){    private-val $(echo SLV_$(slv-repo)_URL    | private-upper ) ; }
slv-info(){  cat << EOI

  Name of the master repo and credentials to contact it with 

   slv-repo   : $(slv-repo)
   slv-user   : $(slv-user)
   slv-pass   : $(slv-pass)
   slv-url    : $(slv-url)
   slv-builds : $(slv-builds)

EOI
}


slv-nginx(){
   local msg="=== $FUNCNAME :"
   echo $msg symbolic link into the logs directory into nginx-htdocs with : nginx-ln \$DYB/logs
   echo $msg incorporate smth like the below into nginx config using nginx-edit 
   slv-nginx-
}

slv-nginx-(){ cat << EOX

     ## $FUNCNAME  

     default_type   text/plain;

     location /logs {
           autoindex on ;
           autoindex_exact_size off ;
           autoindex_localtime on ;

           auth_basic "dyblogs" ;
           auth_basic_user_file  users.txt ;

        }


EOX
}


slv-xml(){ cat << EOX
<result duration="6" status="success" step="cmt" time="2010-06-30T11:45:16.688155">
   <log generator="http://bitten.cmlenz.net/tools/sh#exec">
      <message level="info">FABRICATED BY $FUNCNAME FOR SLAVE TESTING </message>
      <message level="info">Updating existing installation directory installation/trunk/dybinst.</message>
      <message level="info">Updating existing installation directory installation/trunk/dybtest.</message>
      <message level="info">Logging to dybinst-20100630-194520.log (or dybinst-recent.log)</message>
   </log>
</result>
EOX
}

slv-post-(){
  local build=${1-3490}
  local tmp=/tmp/$USER/env/$FUNCNAME/post.xml && mkdir -p $(dirname $tmp)
  slv-xml > $tmp
  cat << EOC
curl -H "Content-Type: application/x-bitten+xml"  --user $(slv-user):$(slv-pass) -d "@$tmp"   $(slv-builds)/$build/steps/   
EOC
}


slv-post(){
   echo $msg 
   slv-post- $*
   eval $(slv-post- $*)
}







slv-sv(){
  local slv=${1:-dybslv}  ## misleading to call this dybinst ... it will perform any config the master directs
  sv-
  slv-sv- $slv | sv-plus $slv.ini
}

slv-sv-(){ cat << EOS
[program:$1]
environment=HOME=$HOME,BITTEN_SLAVE=$(which bitten-slave),SLAVE_OPTS=--verbose
directory=$DYB
command=$DYB/dybinst -l dybinst-slave.log trunk slave
redirect_stderr=true
redirect_stdout=true
autostart=true
autorestart=true
priority=999
user=$USER
EOS
}


slv-recipe-update-all(){

   source $(slv-srcdir)/recipe.bash 
   recipe-update local.dybinst
   recipe-update dybinst
   recipe-update detdesc

}


