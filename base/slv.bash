# === func-gen- : base/slv fgp base/slv.bash fgn slv fgh base
slv-src(){      echo base/slv.bash ; }
slv-source(){   echo ${BASH_SOURCE:-$(env-home)/$(slv-src)} ; }
slv-vi(){       vi $(slv-source) $(slv-slave-path) $(slv-runtest-path) $(slv-isotest-path) ; }

## these are dev versions ...  operational ones now in installation/trunk/dybinst/scripts :w
slv-isotest-path(){ echo $(env-home)/offline/isotest.sh ; }
slv-runtest-path(){ echo $(env-home)/offline/runtest.sh ; }
slv-slave-path(){   echo $(env-home)/offline/slave.sh ; }



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

   == is zeroconf possible ? ==

      * given that credentials cannot be left in the recipe 
      * translate the needed keys from $HOME/.dybinstrc into the 
        ini format needed by the slave at runtime 
           ... so users never need touch the config
           ... do this in a slave runner script  
                                     
   == TODO : ==

       1) propagate NUWA_LOGURL via dybinst local config rather than environment 
          to keep this out of the recipe and config

       2) aiming towards zero-conf ... 

       3) placement of outputs/logs ... caution regards credentials and web access

       4) migrate the update "standard" slave run to use recipes generated here  
         
       5) simplify invokation using a slave.sh script that can be invoked from dybinst
          which derives the config in ini format needed by bitten-slave and requests master
          for a build            
  
              ./dybinst trunk slave

              ./dybinst -s update trunk slave 
              ./dybinst -s green  trunk slave 
              ./dybinst -s shared trunk slave 

          caveats
              * system python needs bitten installed 
              * perhaps with patched extra option from me
          
          downside
              * slave cannot then start from an empty dir  
                    (modulo a dybinst config file providing credentials for the slave )


   == DONE ==

       1) dybinst hookup for test running
               ./dybinst trunk test rootiotest


   == DEV FUNCS ==

     slv--

     slv-cfg-path  : $(slv-cfg-path)
     slv-cfg
           emit config to stdout

     slv-recipe
           emit build/test recipe to stdout 
           normally the recipe is kept on the master ... but convenient to keep all together for dev  
           for svn:export of a file use "dir" attribute for the name of the file 
           current bitten on cms01 doesnt recognize username/password attributes ... so only working due to svn auth cache 

     slv-cmd
           emit cmd to stdout 


     slv-dir : $(slv-dir)
     slv-name : $(slv-name)

       * .recipe resides on the master in normal operation 


    Run inside screen with :
          SCREEN=screen slv--
              (detach with ctrl-a d , re-attach with screen -r ) 
           this builds beneath $(slv-dir)

    To build in the PWD ... eg :
          cd $(local-base)/dyb ; SCREEN=screen slv---

    Follow whats happenin :
          pstree -al $(pgrep -n screen)


EOU
}
slv-env(){      
   elocal-  
   private-
}
slv-cd(){  cd $(slv-dir); }
slv-init(){
   local dir=$(slv-dir) &&  mkdir -p $dir && cd $dir
}

slv-name(){ hostname -s ; }
slv-repo(){        private-val SLV_REPO ; }
slv-repo-builds(){ private-val $(echo SLV_$(slv-repo)_BUILDS | private-upper ) ; }
slv-repo-user(){   private-val $(echo SLV_$(slv-repo)_USER   | private-upper ) ; }
slv-repo-pass(){   private-val $(echo SLV_$(slv-repo)_PASS   | private-upper ) ; }
slv-repo-url(){    private-val $(echo SLV_$(slv-repo)_URL    | private-upper ) ; }
slv-repo-info(){  cat << EOI

  Name of the master repo and credentials to contact it with 

   slv-repo        : $(slv-repo)
   slv-repo-user   : $(slv-repo-user)
   slv-repo-pass   : $(slv-repo-pass)
   slv-repo-url    : $(slv-repo-url)
   slv-repo-builds : $(slv-repo-builds)

EOI
}

slv-dybcnf(){
  local key=${1:-nokey}
  local dybinstrc=$HOME/.dybinstrc
  [ -f "$dybinstrc" ] && . $dybinstrc
  eval local val=\$$key
  echo $val
}




slv-recipe(){ 

  local tmp="local."
  local release=trunk 

  local export=1
  local stages="cmt checkout external"
  local projs="relax gaudi lhcb dybgaudi"
  local testpkgs="gaudimessages gentools rootiotest simhistsexample"
  local xexternals=""

  #local export=1
  #local stages=""
  #local projs=""
  #local testpkgs="rootiotest"
  #local testpkgs="gaudimessages gentools rootiotest simhistsexample dbivalidate"
  #local xexternals=""

  # head
  cat << EOH
<!DOCTYPE build [
  <!ENTITY  nuwa    " export NUWA_LOGURL=\${slv.logurl} ; export BUILD_NUMBER=\${${tmp}build} ; " >
  <!ENTITY  unset   " unset SITEROOT ; unset CMTPROJECTPATH ; unset CMTPATH ; unset CMTEXTRATAGS ; unset CMTCONFIG ; " >
  <!ENTITY  env     " &nuwa; &unset;  " > 

]>
<build
    xmlns:python="http://bitten.cmlenz.net/tools/python"
    xmlns:svn="http://bitten.cmlenz.net/tools/svn"
    xmlns:sh="http://bitten.cmlenz.net/tools/sh"
  >
EOH

  # export
  [ "$export" == "1" ] && cat << EOX
<step id="export" description="export" onerror="fail" >
    <sh:exec executable="bash" output="export.out"      args=" -c &quot; &env; svn export --username \${slv.username} --password \${slv.password} http://dayabay.ihep.ac.cn/svn/dybsvn/installation/trunk/dybinst/dybinst  &quot; " /> 
</step>
EOX

  # stages
  local stage ; for stage in $stages ; do 
  cat << EOS
<step id="$stage" description="$stage" onerror="fail" > 
    <sh:exec executable="bash" output="$stage.out"      args=" -c &quot; &env; ./dybinst -z \${${tmp}revision} \${nuwa.release} $stage &quot; " /> 
</step>  
EOS
  done

  # xexternals 
  local xext ; for xext in $xexternals ; do 
  cat << EOS
<step id="$xext" description="$xext" onerror="continue" > 
    <sh:exec executable="bash" output="$xext.out"      args=" -c &quot; &env; ./dybinst \${nuwa.release} external $xext &quot; " /> 
</step>  
EOS
  done

  # projs
  local proj ; for proj in $projs ; do 
  cat << EOP
<step id="$proj" description="$proj" onerror="fail" > 
    <sh:exec executable="bash" output="$proj.out"  args=" -c &quot; &env; ./dybinst \${nuwa.release} projects $proj  &quot; " /> 
</step>  
EOP
  done

  # testpkgs
  local pkg ; for pkg in $testpkgs ; do 
  cat << EOT
<step id="test-$pkg" description="test-$pkg" onerror="continue" >
     <sh:exec executable="bash"  output="test-$pkg.out" args=" -c &quot;  &env; ./dybinst -m \${${tmp}path} \${nuwa.release} tests $pkg  &quot;  " /> 
     <python:unittest file="test-$pkg.xml" />
</step>
EOT
  done  

  # tail
  cat << EOT
</build>
EOT
}





slv-cmd(){
  local arg=$1  
  cat << EOC
$SCREEN $(which bitten-slave)
      --dry-run
      --config=$(slv-cfg-path)
      --verbose 
      --keep-files 
      --log=$(slv-label).log 
      --user=$(slv-repo-user) --password=$(slv-repo-pass) 
EOC
  slv-layout-$(slv-mode)
  cat << EOA
      $arg
EOA
}

slv-label(){ echo $(slv-repo)-$(slv-mode) ; }

slv-sv-(){ cat << EOS
[program:$(slv-label)]
directory=$(slv-dir)
command=$(which python) $(slv-cmd $*)
redirect_stderr=true
redirect_stdout=true
autostart=true
autorestart=true
priority=999
user=$USER
EOS
}




slv---(){

  # in normal usage the recipe comes from the master 
  # and the config is fixed once ...  
  #
  local msg="=== $FUNCNAME : "
  echo $msg running build recipe from $PWD 

  local path=$(slv-cfg-path) 
  mkdir -p $(dirname $path)
  slv-cfg > $path 

  local recipe="recipe.xml" 
  slv-recipe > $recipe
  xmllint --noout $recipe
  [ "$?" != "0" ] && echo invalid recipe xml $recipe && return 1 

  #local cmd=$(slv-cmd $(slv-repo-builds))    ## remote builds url 
  local cmd=$(slv-cmd $recipe)                ## local recipe for dev
  
  echo $msg $cmd 
  eval $cmd 
}


slv--(){
  local msg="=== $FUNCNAME : "
  [ ! -d "$(slv-dir)" ] && slv-init
  slv-cd
  slv---
}

slv-runtest-demo(){
  cd $DYB
  ./dybinst -m / trunk tests
}

slv-slave-demo(){
  cd $DYB
  $(slv-slave-path) $*

}


slv-screen(){
  cd $DYB || return 1
  SCREEN=screen slv--- 
}



