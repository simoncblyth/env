
scm-usage(){
  
   cat << EOU

      An SCM is a combination of an SVN repository and a Tracitory , this provides
      commands to create these two tied objects in a coherent way.


      scm-create  <name> <arg> 
               
           Create a repository+tracitory (an scm) named <name> and if <arg> is 
           a valid directory path then import the content into it
           if <arg> is INIT then just create the branches/tags/trunk or if EMPTY
           the default leave the repository at revision 0

   
EOU

}

scm-env(){
  elocal-
}


scm-create(){

  svn-
  svn-create $*
  
  trac-
  trac-create $*
   

}