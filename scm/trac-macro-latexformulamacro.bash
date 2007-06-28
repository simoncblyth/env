






trac-macro-latexformulamacro-install(){ 

   ## http://trac-hacks.org/wiki/LatexFormulaMacro
   ## invoke with sudo -u $APACHE2_USER bash -lc "trac-macro-latexformulamacro-install"
 
   local name=${1:-dummy}
   [ "$name" == "dummy" ] && echo must provide the name of the environment && return 1
   
   local fold=$SCM_FOLD/tracs/$name
   [ -d "$fold" ] || ( echo trac-macro-latexformulamacro-install error no folder $fold && exit 1 )
     
   cd $fold/wiki-macros  
   if [ -f "formula.py" ]; then
     svn up formula.py
   else  
     svn co http://trac-hacks.org/svn/latexformulamacro/0.9/formula.py
   fi
   
   local tmp=$SCM_FOLD/tmp
   mkdir -p $tmp 

   ini-edit $fold/conf/trac.ini latex:temp_dir:$tmp latex:fleqn:0 latex:fleqn_width:5% 

}