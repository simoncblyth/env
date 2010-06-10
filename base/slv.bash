# === func-gen- : base/slv fgp base/slv.bash fgn slv fgh base
slv-src(){      echo base/slv.bash ; }
slv-source(){   echo ${BASH_SOURCE:-$(env-home)/$(slv-src)} ; }
slv-vi(){       vi $(slv-source) ; }
slv-env(){      elocal- ; }
slv-usage(){
  cat << EOU
     slv-src : $(slv-src)
     slv-dir : $(slv-dir)

   bitten slave explorations ...

     simon:e blyth$ python -c "import bitten ; print bitten.__file__ "
/usr/local/env/trac/package/bitten/trac-0.11/bitten/__init__.pyc 


   when exporting a file the "dir" is the name of the file 

   <svn:export url="http://dayabay.phys.ntu.edu.tw/repos/env/" 
              path="trunk/env.bash" 
          revision="${revision}" 
               dir="env.bash" />

   if it is the name of an existing directory eg /tmp,  then svn returns errors and writes tempfiles into /tempfile.2.tmp

       [DEBUG   ] Executing ['svn', 'export', '--force', '-r', '2900', 'http://dayabay.phys.ntu.edu.tw/repos/env/trunk/env.bash', '/tmp']
       [ERROR   ] svn: Can't move '/tempfile.2.tmp' to '/tmp': Permission denied
       [DEBUG   ] svn exited with code 256


EOU
}
slv-dir(){ echo $(local-base)/env/base/slv ; }
slv-cd(){  cd $(slv-dir); }
slv-mate(){ mate $(slv-dir) ; }
slv-init(){
   local dir=$(slv-dir) &&  mkdir -p $dir && cd $dir
}

slv-master(){  echo http://dayabay.phys.ntu.edu.tw/tracs/env/builds ; }
slv-name(){ hostname -s ; }
slv-cmd(){  
  local name=$(slv-name)
  private-
  cat << EOC
$(which bitten-slave) --dry-run --name $name --verbose --work-dir=$(slv-dir) --keep-files --log=$name.log --user=$(private-val SLV_USER) --password=$(private-val SLV_PASS) $(slv-master)
EOC
}

slv--(){
  local msg="=== $FUNCNAME : "
  local cmd=$(slv-cmd)
  echo $msg $cmd
  [ ! -d "$(slv-dir)" ] && slv-init
  slv-cd
  eval $cmd
}



