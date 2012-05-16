navadd-src(){    echo trac/package/navadd.bash ; }
navadd-source(){ echo ${BASH_SOURCE:-$(env-home)/$(navadd-source)} ; }
navadd-vi(){     vi  $(navadd-source) ; }


navadd-usage(){
   package-usage  ${FUNCNAME/-*/}
   cat << EOU
  
      http://trac-hacks.org/wiki/NavAddPlugin
      http://trac-hacks.org/browser/navaddplugin/0.9/navadd/navadd.py

     This allows to add buttons to the navbar ... eg "MyTickets"
     Unfortunately it appears that only dumb server relative or absolute URLs work ...

     [components] 
     navadd.* = enabled 
    
                                
EOU

}


navadd-notes(){

cat << EON

testing manual addition ...

   navadd-conf-manual-

to control button ordering need entry in trac/mainnav:: 

	[trac] 
	...
	; mainnav added for control of docs button position
	mainnav = wiki,timeline,roadmap,browser,report,newticket,search,tags,admin,blog,build,query,docs






EON
}

navadd-conf-manual-(){


cat << EOC
[navadd]
add_items = query,docs

query.title = Query
query.url = /tracs/$TRAC_INSTANCE/query
query.perm = REPORT_VIEW
query.target = mainnav   # metanav

docs.title = Docs
docs.url = /${TRAC_INSTANCE:0:1}docs
docs.perm = REPORT_VIEW
docs.target = mainnav   # metanav

EOC
}


navadd-conf-(){
cat << EOC
[navadd]
add_items = query,docs
query.title = Query
query.url = /tracs/$TRAC_INSTANCE/query
query.perm = REPORT_VIEW
query.target = mainnav   # metanav

EOC
}

navadd-triplets(){ 
   local  name=${1:-query}
   local title=${2:-Query}
   local   url=${3:-/tracs/$TRAC_INSTANCE/query}
   cat << EOT
      navadd:add_items:$name
      navadd:$name.title:$title
      navadd:$name.url:$url
      navadd:$name.perm:REPORT_VIEW
      navadd:$name.target:mainnav 
EOT
}

navadd-env(){
  elocal-
  package-
  export NAVADD_BRANCH=0.9
}

navadd-revision(){  echo 6038 ; }
navadd-url(){       echo http://trac-hacks.org/svn/navaddplugin/0.9 ; }
navadd-package(){   echo navadd ; }

navadd-fix(){
   local msg="=== $FUNCNAME :"
   cd $(navadd-dir)   
   echo no fixes
}

navadd-perms(){

 local msg="=== $FUNCNAME :"
 echo $msg 

}


navadd-prepare(){
   navadd-enable $*
   navadd-perms $*
}

navadd-makepatch(){  package-fn $FUNCNAME $* ; }
navadd-applypatch(){ package-fn $FUNCNAME $* ; }

navadd-branch(){    package-fn $FUNCNAME $* ; }
navadd-basename(){  package-fn $FUNCNAME $* ; }
navadd-dir(){       package-fn $FUNCNAME $* ; } 
navadd-egg(){       package-fn $FUNCNAME $* ; }
navadd-get(){       package-fn $FUNCNAME $* ; }

navadd-install(){   package-fn $FUNCNAME $* ; }
navadd-uninstall(){ package-fn $FUNCNAME $* ; }
navadd-reinstall(){ package-fn $FUNCNAME $* ; }
navadd-enable(){    package-fn $FUNCNAME $* ; }

navadd-status(){    package-fn $FUNCNAME $* ; }
navadd-auto(){      package-fn $FUNCNAME $* ; }
navadd-diff(){      package-fn $FUNCNAME $* ; } 
navadd-rev(){       package-fn $FUNCNAME $* ; } 
navadd-cd(){        package-fn $FUNCNAME $* ; }

navadd-fullname(){  package-fn $FUNCNAME $* ; }
navadd-update(){    package-fn $FUNCNAME $* ; }





