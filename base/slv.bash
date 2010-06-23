# === func-gen- : base/slv fgp base/slv.bash fgn slv fgh base
slv-src(){      echo base/slv.bash ; }
slv-source(){   echo ${BASH_SOURCE:-$(env-home)/$(slv-src)} ; }
slv-vi(){       vi $(slv-source) $(slv-runtest-path) $(slv-isotest-path) ; }
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
   
                                     
   == TODO : ==

       1) propagate NUWA_LOGURL via dybinst local config rather than environment 
          to keep this out of the recipe and config

       2) aiming towards zero-conf ... 

       3) migrate the update "standard" slave run to use recipes generated here  
          would it make sense to ...
              ./dybinst trunk slave 


   == DONE ==

       1) dybinst hookup ...
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

slv-cfg-path(){ echo $HOME/.bitten-slave/$(slv-repo).cfg ; }
slv-cfg(){ cat << EOC
#
#  config is available in the recipe context of the slave as repo.url etc..
#     NB this config is entirely relative and thus a single config should work fine for 
#        multiple installations under testing  
#
# normally the master provides this context ...  hardcode under local. for testing
[local]
path = /
build = 1000
config = dybinst
revision = 8799

[repo]
url  = $(slv-repo-url)
user = $(slv-repo-user)
pass = $(slv-repo-pass)

[script]
path = installation/trunk/dybinst/dybinst 
name = dybinst

[nuwa]
release = trunk
logurl = http://localhost:2020

EOC
}

slv-recipe(){ 

  local tmp="local."

  #local export=1
  #local stages="cmt checkout external"
  #local projs="relax gaudi lhcb dybgaudi"
  #local testpkgs="gaudimessages gentools rootiotest simhistsexample dbivalidate"
  #local xexternals=""

  local export=1
  local stages=""
  local projs=""
  #local testpkgs="rootiotest"
  local testpkgs="gaudimessages gentools rootiotest simhistsexample dbivalidate"
  local xexternals=""

  # head
  cat << EOH
<!DOCTYPE build [
  <!ENTITY  nuwa    " export NUWA_LOGURL=\${nuwa.logurl} ; " >
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
<step id="export" description="export \${repo.script}  " onerror="fail" >
    <svn:export url="\${repo.url}" path="\${script.path}" dir="\${script.name}" revision="\${${tmp}revision}" /> 
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
     <sh:exec executable="bash"  output="test-$pkg.out" args=" -c &quot;  &env; ./dybinst -m \${${tmp}path}  \${nuwa.release} tests $pkg  &quot;  " /> 
     <python:unittest file="test-$pkg.xml" />
</step>
EOT
  done  

  # tail
  cat << EOT
</build>
EOT
}


slv-mode(){ 
  echo ${SLV_MODE:-update}   ## update/shared/green/dev
}
slv-layout-dev(){ slv-layout-update ; }
slv-layout-update(){ cat << EOL
    --work-dir=. 
    --build-dir=. 
EOL
}
slv-layout-shared(){ cat << EOL
   --work-dir=. 
   --build-dir="shared_\\\${${tmp}config}_\\\${${tmp}revision}" 
EOL
}
slv-layout-green(){ cat << EOL
   --work-dir=. 
   --build-dir="green_\\\${${tmp}config}_\\\${${tmp}revision}" 
EOL
}

slv-base(){ echo $(local-base)/slv ; }
slv-dir(){ 
  case $(slv-mode) in 
    shared|dev|green) echo $(slv-base)/$(slv-mode) ;;
              update) echo $(slv-base) ;;
  esac
}

slv-external(){
  case $(slv-mode) in 
   green|update) echo -n          ;;   ## default external location in the dybinst dir
     dev|shared) echo ../external ;;   ## one up to the update external 
  esac 
}




slv-cmd(){
  local arg=$1  
  cat << EOC
$SCREEN $(which bitten-slave)
      --dry-run
      --config=$(slv-cfg-path)
      --verbose 
      --keep-files 
      --log=$(slv-name).log 
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
  xmllint $recipe
  [ "$?" != "0" ] && echo invalid recipe xml && return 1 

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

## these are dev versions ...  operational ones now in installation/trunk/dybinst/scripts :w
slv-isotest-path(){ echo $(env-home)/offline/isotest.sh ; }
slv-runtest-path(){ echo $(env-home)/offline/runtest.sh ; }

slv-runtest-demo(){
  cd $DYB
  ./dybinst -m / trunk tests
}






