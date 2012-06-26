
tracxsltmacro-usage(){ cat << EOU

EOU
}

tracxsltmacro-get(){

   # http://www.trac-hacks.org/wiki/XsltMacro

   # cd $LOCAL_BASE/trac
   # [ -d "plugins" ] || mkdir -p plugins
   # cd plugins

   cd $HOME

   local macro=xsltmacro-blyth
   local base=${macro/-blyth/}    ## bash pattern substitution 
  
  if [ -d "$macro" ]; then
     echo tracxsltmacro-get ERROR folder $macro exists already 
     return 1
  fi
  
 
   svn co http://dayabay.phys.ntu.edu.tw/repos/tracdev/$base/branches/$macro  $macro     

    ## the original was from:    http://trac-hacks.org/svn/xsltmacro/0.9/ 
}

tracxsltmacro-update(){

  local macro=xsltmacro-blyth
  #local dir=$LOCAL_BASE/trac/plugins/$macro
  local dir=$HOME/$macro
  
  local cmd="svn update $dir "
  
  
  
  
  if [ -d $dir ]; then
      echo ===== tracxsltmacro-update proceededing to: $cmd  
  else
      echo tracxsltmacro-update ERROR $dir does not exist 
      return 1 
  fi

  eval $cmd 

}



tracxsltmacro-install(){
   
   ## for a reinstallation after local changes to the source distro
   
   local def_macro=xsltmacro-blyth
   local macro=${1:-$def_macro}
   # cd $LOCAL_BASE/trac/plugins/$macro
   cd $HOME/$macro
   
   $SUDO easy_install -Z .


#
# sudo easy_install -Z .
# Password:
# Processing .
# Running setup.py -q bdist_egg --dist-dir /Users/blyth/xsltmacro-netjunki/egg-dist-tmp-UfuN5n
# zip_safe flag not set; analyzing archive contents...
# Adding XsltMacro 0.7 to easy-install.pth file
#
# Installed /Library/Python/2.5/site-packages/XsltMacro-0.7-py2.5.egg
# Processing dependencies for XsltMacro==0.7
# Finished processing dependencies for XsltMacro==0.7
#
#
#
#
#
# simon:xsltmacro-blyth blyth$ sudo easy_install -Z .
# Password:
# Processing .
# Running setup.py -q bdist_egg --dist-dir /Users/blyth/xsltmacro-blyth/egg-dist-tmp-stRa-P
# zip_safe flag not set; analyzing archive contents...
# Adding xslt 0.6 to easy-install.pth file
#
# Installed /Library/Python/2.5/site-packages/xslt-0.6-py2.5.egg
# Processing dependencies for xslt==0.6
# Finished processing dependencies for xslt==0.6
# simon:xsltmacro-blyth blyth$ l  /Library/Python/2.5/site-packages/
# total 1784
# -rw-r--r--   1 root   wheel  168975 Jan 22 22:38 Genshi-0.4.4-py2.5.egg
# drwxr-xr-x   4 root   admin     136 Feb 20 19:32 Genshi-0.5dev_r800-py2.5-macosx-10.5-ppc.egg
# drwxr-xr-x   4 root   admin     136 Jan 22 22:35 Pygments-0.9-py2.5.egg
# -rw-rw-r--   1 root   admin     119 Oct  6 12:11 README
# drwxr-xr-x   4 root   admin     136 Jan 22 22:41 Trac-0.11b1-py2.5.egg
# drwxr-xr-x   4 root   admin     136 Feb 20 22:32 TracAccountManager-0.2dev_r3111-py2.5.egg
# drwxr-xr-x   4 root   admin     136 Feb 20 20:11 TracNav-4.0pre6-py2.5.egg
# drwxr-xr-x   4 root   admin     136 Feb 20 19:32 TracTags-0.6-py2.5.egg
# drwxr-xr-x   4 root   admin     136 Feb 20 23:42 TracTocMacro-11.0.0.2-py2.5.egg
# -rw-r--r--   1 root   wheel  353438 Mar  9 17:13 appscript-0.18.1-py2.5-macosx-10.5-ppc.egg
# -rw-r--r--   1 root   admin     504 Mar 10 13:38 easy-install.pth
# drwxr-xr-x   4 root   admin     136 Mar  9 16:41 ipython-0.8.2-py2.5.egg
# -rw-r--r--   1 root   admin      47 Feb 21 19:56 libxml2.pth
# -rw-r--r--   1 root   admin      47 Feb 21 20:34 libxslt.pth
# drwxr-xr-x  39 root   admin    1326 Jan 22 22:12 mod_python
# -rw-r--r--   1 root   admin     254 Jan 22 22:12 mod_python-3.3.1-py2.5.egg-info
# -rw-r--r--   1 blyth  wheel  362606 Mar  9 16:16 readline-2.5.1-py2.5-macosx-10.5-ppc.egg
# drwxr-xr-x   4 root   admin     136 Mar 10 13:38 xslt-0.6-py2.5.egg
#
#
#
#
# Processing .
# Running setup.py -q bdist_egg --dist-dir /usr/local/trac/plugins/xsltmacro/egg-dist-tmp-Ya8Z0B
# zip_safe flag not set; analyzing archive contents...
# Adding xslt 0.6 to easy-install.pth file
#
# Installed /usr/local/python/Python-2.5.1/lib/python2.5/site-packages/xslt-0.6-py2.5.egg
# Processing dependencies for xslt==0.6
# Finished processing dependencies for xslt==0.6
#
#
# Processing .
# Running setup.py -q bdist_egg --dist-dir /data/usr/local/trac/plugins/xsltmacro/egg-dist-tmp-3SrfA1
# zip_safe flag not set; analyzing archive contents...
# Adding xslt 0.6 to easy-install.pth file
#
# Installed /data/usr/local/python/Python-2.5.1/lib/python2.5/site-packages/xslt-0.6-py2.5.egg
# Processing dependencies for xslt==0.6
#
#

}


tracxsltmacro-test(){

  python -c "import xslt"

#   fails on g4pb , succeeds on hfag
#
#
# Traceback (most recent call last):
#  File "<string>", line 1, in <module>
#  File "xslt/__init__.py", line 2, in <module>
#    from Xslt import *
#  File "xslt/Xslt.py", line 64, in <module>
#    import libxml2
# ImportError: No module named libxml2  
#
# hmm on g4pb do not yet have libxml2 ... despite having lxml
#   after using the standard approach to building libxml2 and libxslt and python bindings this succeeds on G4PB   
  
}


tracxsltmacro-enable(){

   local name=${1:-$SCM_TRAC}
   ini-edit $SCM_FOLD/tracs/$name/conf/trac.ini components:xslt.\*:enabled

}


tracxsltmacro-propagate-env(){
 
   local name=${1:-$SCM_TRAC}
   local ini=$SCM_FOLD/tracs/$name/conf/trac.ini
   
   if [ ! -f $ini ]; then
        echo tracxsltmacro-propagate-env error no such trac.ini file $ini  
        return 1 
   fi
   
   local vars="APACHE_LOCAL_FOLDER APACHE_MODE HFAG_PREFIX"

   for var in $vars
   do 
      eval vval=\$$var
      if [ "X$vval" == "X" ]; then
         echo tracxsltmacro-propagate-env error not defined $var
      else   
         local cmd="ini-edit $SCM_FOLD/tracs/$name/conf/trac.ini xslt:$var:$vval"
         echo $cmd
         eval $cmd 
      fi 
   done
         
}
