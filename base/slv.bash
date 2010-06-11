# === func-gen- : base/slv fgp base/slv.bash fgn slv fgh base
slv-src(){      echo base/slv.bash ; }
slv-source(){   echo ${BASH_SOURCE:-$(env-home)/$(slv-src)} ; }
slv-vi(){       vi $(slv-source) ; }
slv-usage(){
  cat << EOU
     slv-src : $(slv-src)
     slv-dir : $(slv-dir)

     slv-name : $(slv-name)
             using simple short hostname for the slave names
              NB this is different names from current bitten slave 
                  in order to prevent interference 
              ... config the allowed slave names in the master eg for config dybdaily 


   bitten slave explorations ...

     simon:e blyth$ python -c "import bitten ; print bitten.__file__ "
/usr/local/env/trac/package/bitten/trac-0.11/bitten/__init__.pyc 


   when using svn:export for a file the "dir" is the name of the file 

   <svn:export url="http://dayabay.phys.ntu.edu.tw/repos/env/" 
              path="trunk/env.bash" 
          revision="${revision}" 
               dir="env.bash"
            
       />

   if it is the name of an existing directory eg /tmp,  then svn returns errors and writes tempfiles into /tempfile.2.tmp

       [DEBUG   ] Executing ['svn', 'export', '--force', '-r', '2900', 'http://dayabay.phys.ntu.edu.tw/repos/env/trunk/env.bash', '/tmp']
       [ERROR   ] svn: Can't move '/tempfile.2.tmp' to '/tmp': Permission denied
       [DEBUG   ] svn exited with code 256


EOU
}
slv-env(){      
   elocal-  
   private-
}
slv-dir(){ echo $(local-base)/env/base/slv ; }
slv-cd(){  cd $(slv-dir); }
slv-mate(){ mate $(slv-dir) ; }
slv-init(){
   local dir=$(slv-dir) &&  mkdir -p $dir && cd $dir
}

slv-name(){ hostname -s ; }

slv-repo(){        private-val SLV_REPO ; }
slv-repo-builds(){ private-val $(echo SLV_$(slv-repo)_BUILDS | private-upper ) ; }
slv-repo-user(){   private-val $(echo SLV_$(slv-repo)_USER   | private-upper ) ; }
slv-repo-pass(){   private-val $(echo SLV_$(slv-repo)_PASS   | private-upper ) ; }
slv-repo-url(){    private-val $(echo SLV_$(slv-repo)_URL   | private-upper ) ; }
slv-repo-script(){      private-val $(echo SLV_$(slv-repo)_SCRIPT     | private-upper ) ; }
slv-repo-scriptname(){  private-val $(echo SLV_$(slv-repo)_SCRIPTNAME | private-upper ) ; }
slv-repo-info(){  cat << EOI

   slv-repo        : $(slv-repo)
   slv-repo-user   : $(slv-repo-user)
   slv-repo-pass   : $(slv-repo-pass)
   slv-repo-url    : $(slv-repo-url)
   slv-repo-builds : $(slv-repo-builds)
   slv-repo-script : $(slv-repo-script)
   slv-repo-scriptname : $(slv-repo-scriptname)

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
config = dybinst
revision = 8751

[repo]
name = $(slv-repo)
url  = $(slv-repo-url)
user = $(slv-repo-user)
pass = $(slv-repo-pass)
script = $(slv-repo-script)
scriptname = $(slv-repo-scriptname)
EOC
}

slv-recipe-(){ cat << EOR
<build
    xmlns:python="http://bitten.cmlenz.net/tools/python"
    xmlns:svn="http://bitten.cmlenz.net/tools/svn"
    xmlns:sh="http://bitten.cmlenz.net/tools/sh"
  >
<step id="export" description="export \${repo.script} script from \${repo.name} " onerror="fail" >
    <svn:export url="\${repo.url}" path="\${repo.script}" dir="\${repo.scriptname}" revision="\${local.revision}" /> 
</step>


 <!--
   normally the recipe is kept on the master ... 
 -->
</build>
EOR

# probably need to update bitten on cms01 as export doesnt recognize username ... its using the svn auth cache
#  username="\${repo.user}" password="\${repo.pass}"  

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
$(which bitten-slave) $(slv-opt) 
      --name $name 
      --config=$(slv-cfg-path)
      --verbose 
      --work-dir=$(slv-dir) 
      --build-dir="build_\\\${${tmp}config}_\\\${${tmp}revision}" 
      --keep-files 
      --log=$name.log 
      --user=$(slv-repo-user) --password=$(slv-repo-pass) 
      $(slv-recipe-path) 
EOC
#  normally the command targets  the trac instances url : $(slv-repo-builds)
#  but for recipe development is handy to target a local recipe   

}

slv--(){
  local msg="=== $FUNCNAME : "
  local cmd=$(slv-cmd)
  echo $msg $cmd
  [ ! -d "$(slv-dir)" ] && slv-init

  slv-cd

  slv-cfg 
  slv-recipe

  eval $cmd
}



