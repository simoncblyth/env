fullblog-src(){    echo trac/package/fullblog.bash ; }
fullblog-source(){ echo ${BASH_SOURCE:-$(env-home)/$(fullblog-source)} ; }
fullblog-vi(){     vi  $(fullblog-source) ; }
fullblog-usage(){
   package-usage  ${FUNCNAME/-*/}
   cat << EOU
  
      http://trac-hacks.org/wiki/FullBlogPlugin 
      http://trac-hacks.org/browser/fullblog

   == installation ==

   Setting up an instance to use fullblog...   
          
     1) set the trac config and permissions
         trac-
         fullblog-
         SUDO=sudo TRAC_INSTANCE=newtest fullblog-prepare  
     
     2) check status of http://localhost/tracs/newtest
        should inform that upgrade needed
        (after an apache bounce)
        
     3) perform the upgrade with
          TRAC_INSTANCE=newtest trac-admin-- upgrade 
        
     4) check again  http://localhost/tracs/newtest
        there should be an extra "Blog" tab if you are logged in 
        as a user with permission to see it 

    == usage notes ==

      * create the 'About' blog entry first ... for it to show up as first entry 
      * new post link is underneath the Blog tab 
      * visibility of Comments is not very high 
          * ... seems not configurable 

      * admin > blog > settings   
          * number of posts on front page
          * username and time substitution available, like $USER-%Y/%m/%d/my_topic
          * sidebar test

      * the template just prefils the shortname ... but keep it simple else ugly URLs and TracLinks
          * http://localhost/tracs/newtest/blog/blyth-2010/05/20/my_topic
          * blog:blyth-2010/05/20/my_topic 
       
          * putting dates in these URLs is obnoxious and duplicitous ... with only slight advantage of avoiding clashes
             * adopt convention of url friendly hyphenated descritive phrase without date or name 

     * you can browse by author, time or category  ... so i prefer to keep simple shortnames for posts 
         * http://localhost/tracs/newtest/blog/author/blyth
         * http://localhost/tracs/newtest/blog/2010/5
         * http://localhost/tracs/newtest/blog/category/Test

     * blog entries show up in search results
         * http://localhost/tracs/newtest/tags?q=%27Test%27

     * how to decide : wiki page OR blog entry ?
         * to what degree is the interest in the topic associated with time 
         * the difference beween wikipages and blog entries is that the later form a natural sequence and
           can easily be referenced by date 
         * wikipages are mostly referenced by name, and are updated effectively invisibly  

 
                                
EOU

}


fullblog-notes(){

cat << EON

  On G 
     has to not macports to get the system python but then meet  svn/setuptools incompatibility ...
     really need to move to virtual python env for Trac ... as site-packages is horribly bloated 

{{{
simon:~ blyth$ fullblog-install
=== package-install : fullblog
svn: This client is too old to work with working copy '/usr/local/env/trac/package/fullblog/0.11'; please get a newer Subversion client
}}}

    After blowing away the old 0.11 and re-getting with the old system svn : 

{{{
=== package-install : fullblog
=== package-applypatch : there is no patch file /Users/blyth/env/trac/patch/fullblog/fullblog-0.11-7966.patch
no fixes
Processing .
Running setup.py -q bdist_egg --dist-dir /usr/local/env/trac/package/fullblog/0.11/egg-dist-tmp-hFsN0R
Adding TracFullBlogPlugin 0.1.1-r7774 to easy-install.pth file
Installed /Library/Python/2.5/site-packages/TracFullBlogPlugin-0.1.1_r7774-py2.5.egg
}}}



EON
}



fullblog-env(){
  elocal-
  package-
  export FULLBLOG_BRANCH=0.11
}

fullblog-revision(){  echo 7966 ; }
fullblog-url(){       echo http://trac-hacks.org/svn/fullblogplugin/$(fullblog-branch) ; }
fullblog-package(){   echo tracfullblog ; }

fullblog-fix(){
   local msg="=== $FUNCNAME :"
   cd $(fullblog-dir)   
   echo no fixes
}

fullblog-perms(){

 local msg="=== $FUNCNAME :"
 echo $msg for consistency these are now done in trac/tracperm.bash  
 
 trac-admin-- permission add blyth BLOG_ADMIN
 trac-admin-- permission add authenticated BLOG_VIEW
 trac-admin-- permission add authenticated BLOG_COMMENT
 trac-admin-- permission add authenticated BLOG_MODIFY_OWN
 trac-admin-- permission list

}


fullblog-prepare(){
   fullblog-enable $*
   fullblog-perms $*
}

fullblog-makepatch(){  package-fn $FUNCNAME $* ; }
fullblog-applypatch(){ package-fn $FUNCNAME $* ; }

fullblog-branch(){    package-fn $FUNCNAME $* ; }
fullblog-basename(){  package-fn $FUNCNAME $* ; }
fullblog-dir(){       package-fn $FUNCNAME $* ; } 
fullblog-egg(){       package-fn $FUNCNAME $* ; }
fullblog-get(){       package-fn $FUNCNAME $* ; }

fullblog-install(){   package-fn $FUNCNAME $* ; }
fullblog-uninstall(){ package-fn $FUNCNAME $* ; }
fullblog-reinstall(){ package-fn $FUNCNAME $* ; }
fullblog-enable(){    package-fn $FUNCNAME $* ; }

fullblog-status(){    package-fn $FUNCNAME $* ; }
fullblog-auto(){      package-fn $FUNCNAME $* ; }
fullblog-diff(){      package-fn $FUNCNAME $* ; } 
fullblog-rev(){       package-fn $FUNCNAME $* ; } 
fullblog-cd(){        package-fn $FUNCNAME $* ; }

fullblog-fullname(){  package-fn $FUNCNAME $* ; }
fullblog-update(){    package-fn $FUNCNAME $* ; }





