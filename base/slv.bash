# === func-gen- : base/slv fgp base/slv.bash fgn slv fgh base
slv-src(){      echo base/slv.bash ; }
slv-source(){   echo ${BASH_SOURCE:-$(env-home)/$(slv-src)} ; }
slv-vi(){       vi $(slv-source) ; }
slv-usage(){
  cat << EOU

    == slv ==

        simplify the slave- funcs ...  with an eye to doing daily builds under slave control
           a) work in fully relative manner ... 
           b) minimize config 

           c) do it without the dyb__ funcs ... that are single install fixated
           d) environment isolation ... do it in env -i for insensitivity to calling environment


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



   == fixed interval bitten-slave operation ==

      IDEA :
         * use single shot bitten-slave invokation
         * make the requests to the master at the "right" time based on timeline observations 
              * calling at fixed times would not work if coincides with commit cooldown windows
         * observe the timeline rss feed to time the request
              * http://www.feedparser.org/
 

     slv-dir : $(slv-dir)
     slv-name : $(slv-name)

       * .cfg must not be web-accessible so cannot live in the build dir 
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
slv-dir(){ echo $(local-base)/env/base/slv ; }
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
# config is available in the recipe context of the slave as repo.url etc..

# normally the master provides this context ... 
# hardcode under local. for testing
[local]
path = /
build = 1000
config = dybinst
revision = 8751

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



slv-testdir(){
   case ${1} in
      gaudimessages) echo dybgaudi/Utilities/GaudiMessages ;;
           gentools) echo dybgaudi/Simulation/GenTools  ;;
         rootiotest) echo dybgaudi/RootIO/RootIOTest ;;
          simhistex) echo tutorial/Simulation/SimHistsExample ;;
        dbivalidate) echo dybgaudi/Database/DbiValidate ;;
   esac  
}



slv-recipe(){ 

  local tmp="local."

  #local export=1
  #local stages="cmt checkout external"
  #local projs="relax gaudi lhcb dybgaudi"
  #local tests="gaudimessages gentools rootiotest simhistex dbivalidate"
  #local xexternals=""

  local export=0
  local stages=""
  local projs=""
  local tests="rootiotest"
  local xexternals=""


  cat <<EOC > /dev/null

  <!ENTITY  release " cd \$NUWA_HOME/dybgaudi/DybRelease/cmt ; [ ! -f setup.sh ] &amp;&amp; cmt config ; . setup.sh ; cd .. ;  " >
  <!ENTITY  setup   " cd \$NUWA_HOME/\$NUWA_TESTDIR/cmt      ; [ ! -f setup.sh ] &amp;&amp; cmt config ; . setup.sh ; cd .. ;  " >
  <!ENTITY  testdir " &env; &release; &setup; " >
  <!ENTITY  info    " env | grep BUILD_ ;  " > 
  <!ENTITY  scpt    " . \$PWD/installation/trunk/dybtest/scripts/dyb__.sh ; " > 
  <!ENTITY  nose    " env ; nosetests --with-xml-output --xml-outfile=\$BUILD_PWD/test-\$BUILD_TEST.xml --xml-baseprefix=\$(dyb__relativeto \$BUILD_PATH \$BUILD_CONFIG_PATH)/ ; "  >

}
EOC

  # head
  cat << EOH
<!DOCTYPE build [
  <!ENTITY  base    " export BUILD_PWD=\$PWD ; export BUILD_PATH=\${${tmp}path} ; export BUILD_CONFIG_PATH=\${${tmp}path} ; export BUILD_REVISION=\${${tmp}revision} ; export BUILD_NUMBER=\${${tmp}build} ; " > 
  <!ENTITY  nuwa    " export NUWA_HOME=\$PWD/NuWa-\${nuwa.release} ; export NUWA_LOGURL=\${nuwa.logurl} ; " >
  <!ENTITY  unsite  " unset SITEROOT ; unset CMSPROJECTPATH ; unset CMTPATH ; unset CMTEXTRATAGS ; unset CMTCONFIG ; " >
  <!ENTITY  setup   " . \$NUWA_HOME/setup.sh ; " > 
  <!ENTITY  env     " &base; &nuwa; &unsite;  " > 

  <!ENTITY  test    " env -i NUWA_HOME=\$PWD/NuWa-\${nuwa.release} NUWA_TESTDIR=\$NUWA_TESTDIR $(env-home)/offline/runtest.sh  " > 

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

  # tests
  local tst ; for tst in $tests ; do 
  cat << EOT
<step id="test-$tst" description="test-$tst" onerror="continue" >
     <sh:exec executable="bash"  output="test-$tst.out"
           args=" -c &quot;  export NUWA_TESTDIR=$(slv-testdir $tst) ; &test; &quot; " />
     <python:unittest file="test-$tst.xml" />
</step>
EOT
  done  

  # tail
  cat << EOT
</build>
EOT
}

slv-cmd(){  
  local recipe=${1:-slv.recipe}  
  local name=$(slv-name)
  local tmp="local."
  cat << EOC
$SCREEN $(which bitten-slave)
      --dry-run
      --config=$(slv-cfg-path)
      --verbose 
      --work-dir=. 
      --build-dir="build_\\\${${tmp}config}_\\\${${tmp}revision}" 
      --keep-files 
      --log=$name.log 
      --user=$(slv-repo-user) --password=$(slv-repo-pass) 
      $recipe
EOC
# --name=$name ... default is hostname ? 
}

slv---(){

  local msg="=== $FUNCNAME : "
  echo $msg running build from $PWD wherever that may be 

  local path=$(slv-cfg-path) 
  mkdir -p $(dirname $path)
  slv-cfg > $path 

  local recipe="slv.recipe" 
  slv-recipe > $recipe

  local url=$(slv-repo-builds)   ## remote "builds" url 
  #local cmd=$(slv-cmd $url)
  local cmd=$(slv-cmd)   ## local recipe testing 
  echo $msg $cmd 
  eval $cmd 
}


slv--(){
  local msg="=== $FUNCNAME : "
  [ ! -d "$(slv-dir)" ] && slv-init
  slv-cd
  slv---
}


slv-runtest(){
  local script=$(env-home)/offline/runtest.sh   
  cd $(slv-dir)/build_dybinst_8751 
  env -i NUWA_HOME=$PWD/NuWa-trunk NUWA_TESTDIR=dybgaudi/RootIO/RootIOTest NOSETESTS=$(which nosetests) $script
}



