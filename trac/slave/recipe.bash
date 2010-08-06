recipe-source(){   echo ${BASH_SOURCE} ; }
recipe-(){         . $(recipe-source) ;  }
recipe-vi(){       vi $(recipe-source) ; }
recipe-usage(){
  local config=${1:-dybinst}
  cat << EOU

    Extraction of recipe generation functionality from
        $(env-home)/trac/slave/slv.bash
    into a standalone-ish form, in preparation for integration with dybinst to 
    support the generation of bitten-slave recipes such as "dybinst.xml" with 

       ./dybinst trunk tests recipe:dybinst
       ./dybinst trunk tests recipe:detdesc

     recipe-envpath : $(recipe-envpath)
     recipe-vars    : $(recipe-vars)
     recipe-smry    : list of the vars of interest and their values  

$(recipe-smry)


     Settings for the config $config argument, default dybinst  
     that are used to generate the xml recipe

       recipe-export     $config  : $(recipe-export $config)
       recipe-cmt        $config  : $(recipe-cmt $config)
       recipe-external   $config  : $(recipe-external $config)
       recipe-xexternals $config  : $(recipe-xexternals $config)
       recipe-projs      $config  : $(recipe-projs $config)
       recipe-docs       $config  : $(recipe-docs $config)
       recipe-testpkgs   $config  : $(recipe-testpkgs $config)
      
       recipe-path       $config  : $(recipe-path $config)
            path for the recipe with this config      

       recipe-update     $config
            write the recipe to a temporary file and compare with any existing 
            recipe for this config ... wait for user confirmation to adopt the
            new recipe 

            NB for the recipe to have any effect it must be committed 
               to the repository 

       recipe-recipe     $config  
            emit the xml recipe to stdout


$(recipe-recipe $config)



EOU
}
recipe-srcdir(){   [ -n "$DYB" ] && echo $DYB/installation/trunk/dybinst/scripts || echo $(dirname $(recipe-source)) ; }
recipe-envpath(){   echo $(recipe-srcdir)/dybinst-common.sh ; }
recipe-path(){      echo $(recipe-srcdir)/${1:-dybinst}.xml ; }
recipe-vars(){     env -i bash -c ". $(recipe-envpath) ; echo \${!${1:-dyb_tests}*} " ; }
recipe-lookup(){   env -i bash -c ". $(recipe-envpath) ; echo \$${1:-dyb_tests_default} " ; }
recipe-tests(){    recipe-lookup dyb_tests_${1:-default};  }
recipe-smry(){
   local var
   for var in $(recipe-vars dyb_tests) ; do
      printf "%-20s : %s \n" $var "$(recipe-lookup $var)"
   done
}



recipe-update(){
   local msg="=== $FUNCNAME :"
   [ -z "$DYB" ] && echo $msg DYB is not defined && return 1
   local nam=${1:-dybinst}           ## use local.dybinst  for local variant 
   local cur=$(recipe-path $nam)
   local tmp=/tmp/$USER/env/$FUNCNAME/$nam.xml
   mkdir -p $(dirname $tmp)

   echo $msg writing recipe to $tmp
   recipe-recipe $nam > $tmp
   xmllint --noout $tmp
   [ "$?" != "0" ] && echo invalid recipe xml $tmp && return 1 
 
   if [ -f "$cur" ]; then 
      local cmd="diff $cur $tmp"
      echo $msg $cmd 
      eval $cmd 
   fi
   local ans
   read -p "$msg enter YES to proceed with updating $cur ... remember to check it in "  ans 
   [ "$ans" != "YES" ] && echo $msg skipping && return 0
   cp $tmp $cur
}



recipe-export(){
   case $1 in 
    local*) echo 0 ;;
         *) echo 1 ;;
   esac
}
recipe-cmt(){
  case $1 in 
     *) echo 1 ;;
  esac
}
recipe-checkout(){
  case $1 in 
     *) echo 1 ;;
  esac
}
recipe-external(){
   case $1 in 
 dybdoc) echo 0 ;;
      *) echo 1 ;;
   esac
}
recipe-xexternals(){
  case $1 in
     *) echo -n ;; 
  esac
}
recipe-projs(){
  case $1 in
     *test|*doc) echo -n ;; 
              *) echo $(recipe-lookup dyb_projects) ;;
  esac
}
recipe-docs(){
  case $1 in
     dybinst) echo manual doxyman ;; 
          *) echo -n ;;
  esac
}
recipe-testpkgs(){
  local arg=$1
  [ "${arg:0:6}" == "local." ] && arg=${arg:6} 
  case $arg in
          *) echo $(recipe-lookup dyb_tests_$arg) ;;
  esac
}

recipe-recipe(){ 

  local config=$1
  local tmp=$([ "${config:0:6}" == "local." ] && echo "local." || echo -n )   ## blank for operation with the master

  # head
  cat << EOH
<!DOCTYPE build [
  <!ENTITY  slug        "\${${tmp}config}/\${${tmp}build}_\${${tmp}revision}" >
  <!ENTITY  nuwa        " export BUILD_SLUG=&slug; ; " >
  <!ENTITY  unset       " unset SITEROOT ; unset CMTPROJECTPATH ; unset CMTPATH ; unset CMTEXTRATAGS ; unset CMTCONFIG ; " >
  <!ENTITY  env         " &nuwa; &unset;  " > 
  <!ENTITY  logd        "logs/&slug;" >
  <!ENTITY  dybinst_url "http://dayabay.ihep.ac.cn/svn/dybsvn/installation/trunk/dybinst/dybinst" >
  <!ENTITY  dybinst     "&env; ./dybinst -l &logd;/\${${tmp}config}.log " >

]>
<build
    xmlns:python="http://bitten.cmlenz.net/tools/python"
    xmlns:svn="http://bitten.cmlenz.net/tools/svn"
    xmlns:sh="http://bitten.cmlenz.net/tools/sh"
  >
  <!-- recipe derived by recipe-;$FUNCNAME  for config $config 
       recipe-export     $config  : $(recipe-export $config)
       recipe-cmt        $config  : $(recipe-cmt $config)
       recipe-external   $config  : $(recipe-external $config)
       recipe-xexternals $config  : $(recipe-xexternals $config)
       recipe-projs      $config  : $(recipe-projs $config)
       recipe-docs       $config  : $(recipe-docs $config)
       recipe-testpkgs   $config  : $(recipe-testpkgs $config)
   -->

EOH

   # init 
   cat << EOI
<step id="init" description="init" onerror="fail" >
    <sh:exec executable="bash" output="/dev/null"      args=" -c &quot; &env; mkdir -p &logd;  &quot; " />
</step> 
EOI

  # export
  [ "$(recipe-export $config)" == "1" ] && cat << EOX
<step id="export" description="export" onerror="fail" >
    <sh:exec executable="bash" output="&logd;/export.out"      args=" -c &quot; &env; svn export --username \${slv.username} --password \${slv.password} &dybinst_url; &quot; " /> 
</step>
EOX

  [ "$(recipe-cmt $config)" == "1" ] && cat << EOA
<step id="cmt" description="cmt" onerror="fail" > 
    <sh:exec executable="bash" output="&logd;/cmt.out"      args=" -c &quot; &dybinst;  \${nuwa.release} cmt &quot; " /> 
</step>  
EOA

  [ "$(recipe-checkout $config)" == "1" ] && cat << EOB
<step id="checkout" description="checkout" onerror="fail" > 
    <sh:exec executable="bash" output="&logd;/checkout.out"      args=" -c &quot; &dybinst; -z \${${tmp}revision} \${nuwa.release} checkout &quot; " /> 
</step>  
EOB

  [ "$(recipe-external $config)" == "1" ] && cat << EOC
<step id="external" description="external" onerror="fail" > 
    <sh:exec executable="bash" output="&logd;/external.out"      args=" -c &quot; &dybinst; -c -p  \${nuwa.release} external &quot; " /> 
</step>  
EOC

  # xexternals 
  local xext ; for xext in $(recipe-xexternals $config) ; do 
  cat << EOS
<step id="$xext" description="$xext" onerror="continue" > 
    <sh:exec executable="bash" output="&logd;/$xext.out"      args=" -c &quot; &dybinst; \${nuwa.release} external $xext &quot; " /> 
</step>  
EOS
  done

  # projs
  local proj ; for proj in $(recipe-projs $config) ; do 
  cat << EOP
<step id="$proj" description="$proj" onerror="fail" > 
    <sh:exec executable="bash" output="&logd;/$proj.out"  args=" -c &quot; &dybinst; -c -p \${nuwa.release} projects $proj  &quot; " /> 
</step>  
EOP
  done

  # testpkgs
  local pkg ; for pkg in $(recipe-testpkgs $config) ; do 
  cat << EOT
<step id="test-$pkg" description="test-$pkg" onerror="continue" >
     <sh:exec executable="bash"  output="&logd;/test-$pkg.out" args=" -c &quot; &dybinst; -m \${${tmp}path} \${nuwa.release} tests $pkg  &quot;  " /> 
     <python:unittest file="&logd;/test-$pkg.xml" />
</step>
EOT
  done  

  # docs 
  local doc ; for doc in $(recipe-docs $config) ; do 
  cat << EOP
<step id="$doc" description="$doc" onerror="fail" > 
    <sh:exec executable="bash" output="&logd;/$doc.out"  args=" -c &quot; &dybinst; \${nuwa.release} docs $doc  &quot; " /> 
</step>  
EOP
  done

  # tail
  cat << EOT
</build>
EOT
}


