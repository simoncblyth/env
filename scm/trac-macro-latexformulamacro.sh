#!/bin/bash 
#
#     http://trac-hacks.org/wiki/LatexFormulaMacro 
#  allows writing latex formulas in wiki text, that get converted to images 
#
#  install with the below three steps 
#
#    1) . ~/$ENV_BASE/scm/trac-macro-latexformulamacro.sh 
#    2) trac-macro-latexformulamacro-get 
#    3) sudo -u $APACHE2_USER $HOME/$ENV_BASE/scm/trac-macro-latexformulamacro.sh env
#
#

trac-macro-latexformulamacro-get(){
 
   cd $LOCAL_BASE/trac
   [ -d "wiki-macros" ] || mkdir -p wiki-macros
   cd wiki-macros

   local macro=latexformulamacro
   mkdir -p $macro
   svn co http://trac-hacks.org/svn/$macro/0.9/ $macro

}



trac-macro-latexformulamacro-install(){ 


   local name=${1:-dummy}
   local macro=latexformulamacro
   
   [ "$name" == "dummy" ] && echo must provide the name of the environment && return 1
   
   local fold=$SCM_FOLD/tracs/$name
   [ -d "$fold" ] || ( echo trac-macro-latexformulamacro-install error no folder $fold && exit 1 )
     
   cd $fold/wiki-macros  
   sudo -u $APACHE2_USER cp -f $LOCAL_BASE/trac/wiki-macros/$macro/formula.py .  
     
}

trac-macro-latexformulamacro-conf(){

   local tmp=$SCM_FOLD/tmp
   mkdir -p $tmp 
   ##$HOME/$ENV_BASE/base/ini-edit.pl $fold/conf/trac.ini latex:temp_dir:$tmp latex:fleqn:0 latex:fleqn_width:5% 
   ini-edit  $fold/conf/trac.ini latex:temp_dir:$tmp latex:fleqn:0 latex:fleqn_width:5% 

}


#trac-macro-latexformulamacro-install $*