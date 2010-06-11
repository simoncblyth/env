# === func-gen- : base/slv fgp base/slv.bash fgn slv fgh base
slv-src(){      echo base/slv.bash ; }
slv-source(){   echo ${BASH_SOURCE:-$(env-home)/$(slv-src)} ; }
slv-vi(){       vi $(slv-source) ; }
slv-usage(){
  cat << EOU

    == slv ==

        simplify the slave- funcs to work in fully relative manner 
        with an eye to doing daily builds under slave control

     slv-dir : $(slv-dir)
     slv-name : $(slv-name)

       * .cfg must not be web-accessible so cannot live in the build dir 
       * .recipe resides on the master in normal operation 

    Run inside screen with :
          SCREEN=screen slv--
             (detach with ctrl-a d , re-attach with screen -r ) 
           the default is to do the build beneath $(slv-dir)

    To build in the PWD ... use :
          SCREEN=screen slv---


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


slv-cfg(){
  local msg="=== $FUNCNAME :"
  echo $msg writing $(slv-cfg-path) 
  $FUNCNAME- > $(slv-cfg-path) 
}

slv-cfg-path(){ echo $(slv-repo).cfg ; }
slv-cfg-(){ cat << EOC
# config is available in the recipe context of the slave as repo.url etc..

# normally the master provides this context ... 
# hardcode under local. for testing
[local]
build = 1000
config = dybinst
revision = 8751

[repo]
name = $(slv-repo)
url  = $(slv-repo-url)
user = $(slv-repo-user)
pass = $(slv-repo-pass)

[script]
path = installation/trunk/dybinst/dybinst 
name = dybinst

[dybinst]
release = trunk



EOC
}

slv-recipe-(){ 
  local tmp="local."
  cat << EOR

<!DOCTYPE build [
  <!ENTITY  slav  " export BUILD_REVISION=\${${tmp}revision} ; export BUILD_NUMBER=\${${tmp}build} ; " > 
  <!ENTITY  clean " unset CMTCONFIG ; " >
  <!ENTITY  env   " &slav;  &clean;  " > 
]>

<build
    xmlns:python="http://bitten.cmlenz.net/tools/python"
    xmlns:svn="http://bitten.cmlenz.net/tools/svn"
    xmlns:sh="http://bitten.cmlenz.net/tools/sh"
  >
<step id="export" description="export \${repo.script} script from \${repo.name} " onerror="fail" >
    <svn:export url="\${repo.url}" path="\${script.path}" dir="\${script.name}" revision="\${${tmp}revision}" /> 
</step>
 <!--
     note that the dir is the name of the file 
     update bitten on cms01 as export doesnt recognize username="\${repo.user}" password="\${repo.pass}"  
  -->

<step id="cmt" description="cmt" onerror="fail" > 
    <sh:exec executable="bash" output="cmt.out"      args=" -c &quot; &env; ./dybinst \${dybinst.release} cmt &quot; " /> 
</step>  

<step id="checkout" description="checkout" onerror="fail" > 
    <sh:exec executable="bash" output="checkout.out"  args=" -c &quot; &env; ./dybinst -z \${${tmp}revision} \${dybinst.release} checkout &quot; " /> 
</step>  

<step id="external" description="external" onerror="fail" > 
    <sh:exec executable="bash" output="external.out"  args=" -c &quot; &env; ./dybinst  \${dybinst.release} external &quot; " /> 
</step>  

<step id="relax" description="relax" onerror="fail" > 
    <sh:exec executable="bash" output="relax.out"  args=" -c &quot; &env; ./dybinst \${dybinst.release} projects relax  &quot; " /> 
</step>  

<step id="gaudi" description="gaudi" onerror="fail" > 
    <sh:exec executable="bash" output="gaudi.out"  args=" -c &quot; &env; ./dybinst \${dybinst.release} projects gaudi  &quot; " /> 
</step>  

<step id="lhcb" description="lhcb" onerror="fail" > 
    <sh:exec executable="bash" output="lhcb.out"  args=" -c &quot; &env; ./dybinst \${dybinst.release} projects lhcb  &quot; " /> 
</step>  

<step id="dybgaudi" description="dybgaudi" onerror="fail" > 
    <sh:exec executable="bash" output="dybgaudi.out"  args=" -c &quot; &env; ./dybinst \${dybinst.release} projects dybgaudi  &quot; " /> 
</step>  

 <!-- normally the recipe is kept on the master ... but convenient to keep all together for dev  -->

</build>
EOR


}
slv-recipe-path(){ echo demo.recipe ; }
slv-recipe(){
  local msg="=== $FUNCNAME :"
  echo $msg writing $($FUNCNAME-path) 
  $FUNCNAME- > $($FUNCNAME-path) 
}


slv-opt(){  
  local def="--dry-run"   ## --dry-run is very useful for debugging as avoids having to invalidate the failed builds...  
  #local def=""
  echo ${SLV_OPT:-$def}
}

slv-cmd(){  
  local name=$(slv-name)
  local tmp="local."
  cat << EOC
$SCREEN $(which bitten-slave) $(slv-opt) 
      --name $name 
      --config=$(slv-cfg-path)
      --verbose 
      --work-dir=. 
      --build-dir="build_\\\${${tmp}config}_\\\${${tmp}revision}" 
      --keep-files 
      --log=$name.log 
      --user=$(slv-repo-user) --password=$(slv-repo-pass) 
      $(slv-recipe-path) 
EOC
#  normally the command targets  the trac instances url : $(slv-repo-builds)
#  but for recipe development is handy to target a local recipe   

}

slv---(){

  local msg="=== $FUNCNAME : "
  echo $msg running build from $PWD wherever that may be 
 
  slv-cfg 
  slv-recipe

  local cmd=$(slv-cmd)
  echo $msg $cmd
  eval $cmd
}


slv--(){
  local msg="=== $FUNCNAME : "
  [ ! -d "$(slv-dir)" ] && slv-init
  slv-cd
  slv---
}



