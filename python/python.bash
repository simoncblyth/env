
python-src(){   echo python/python.bash ; }
python-source(){ echo ${BASH_SOURCE:-$ENV_HOME/$(python-src)} ; }
python-vi(){     vi $(python-source) ; }
python-url(){    echo $(env-url)/$python-src ; }
python-syspath(){ python -c "import sys ; print '\n'.join(sys.path) " ; }
python-usage(){ cat << "EOU"

Python
=======


Redirecting logging
---------------------

Logging by default is going to sys.stderr which is not 
so convenient for GPU cluster running which returns
separate .out and .err logs 


* https://stackoverflow.com/questions/16061641/python-logging-split-between-stdout-and-stderr

::

    import logging
    import sys
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


Avoid the pyc
--------------

::

    import sys
    sys.dont_write_bytecode = True


Functions
---------

python-pygments-get
python-crack-egg

python-uninstall <name>
      invoke python-unegg and python-uneasy

python-uneasy <name>  
      remove the name entry from  $PYTHON_SITE/easy-install.pth

python-unegg  <name>
      delete the egg directory from $PYTHON_SITE
    
python-pth
      cat the easy-install.pth 
    
python-isdevelopdir- <dir>
      return code indicates if the passed directory is egglinked into python syspath 
    
python-ls
      ls -l $PYTHON_SITE/
         
python-ldconfig
      does no harm to run this more that once ...
        === env-ldconfig : the dir /data/env/system/python/Python-2.5.1/lib is already there /etc/ld.so.conf
         
python-cd <rel>    
       cd $(python-site)/<rel>  

python-mail 
       echo hello testing python-mail  | python-mail `local-email` 

python-mail-test

        python-mail-test
            Attempting to send email to recipient:[blyth@hep1.phys.ntu.edu.tw] from:[me@localhost] message lines:[4] 

        NB for the mailing to work, a mailer needs to be running, eg on G : *sudo postfix start*




Regexp re.sub
----------------

Doing a replacement that can be switched off with an escape::

    In [30]: re.sub(re.compile('(?:(?P<the_a>[!`]?a)|(?P<the_b>b))', re.UNICODE), lambda m:"["+m.group(0)+"]", " some !ab line ab `a b ")
    Out[30]: ' some [!a][b] line [a][b] [`a] [b] '




Unicode
---------

* https://stackoverflow.com/questions/9942594/unicodeencodeerror-ascii-codec-cant-encode-character-u-xa0-in-position-20
* https://docs.python.org/2.7/howto/unicode.html


Python lib structure
---------------------
          
Currently piecemeal with multiple setup.py scripts 

python-syspath
     dump the syspath


The *qxml/pyextfun* extension uses::

	      python setup.py install --prefix $(ENV_PREFIX)

Writing into  */usr/local/env/lib/python2.5/site-packages/*


Macports Python Selection
--------------------------

::

    sudo port -v install py26-ipython -scientific
    sudo port -v install py26-setuptools
    sudo port -v install py26-nose

    sudo port select --set python python26
    python -V
    sudo port select --set ipython ipython26


GDB Extensions for Python Debugging
--------------------------------------

* https://fedoraproject.org/wiki/Features/EasierPythonDebugging



Running a command and getting output
--------------------------------------


::

    import commands
    print commands.getstatusoutput('wc -l file')



EOU




}



python-mdir(){ 
   local iwd=$PWD
   local tmp=/tmp/$FUNCNAME && mkdir -p $tmp && cd $tmp  ## need to rum from empty dir to avoid module clashes 
   python -c "import $1 as _, os ; print os.path.dirname(_.__file__) " ; 
   cd $iwd
}
python-mate(){ mate $(python-mdir $*) ; }
python-ls(){ ls -l $(python-site)/ ; }

python-srv(){
   python -m SimpleHTTPServer  # serves the invoking directory  on http://localhost:8000
}


python-versions(){
   python -V
   echo ipython $(ipython -V)
}

python-config-vars(){ $FUNCNAME- | python ; }
python-config-vars-(){ cat << EOC
from distutils.sysconfig import get_config_vars 
for k,v in get_config_vars().items():
    pass
    print k,v
    #if "2.5" in str(v):print k,v  
    #if "2.6" in str(v):print k,v  
EOC
}


python-version-system(){  local v=$(python -V 2>&1) ; echo ${v/Python} ;  }
python-version-source(){
   local tag=${1:-$NODE_TAG}
   case $tag in 
     formerC2) echo 2.5.1 ;;
         WW) echo 2.5.6 ;;
     C2|C2R) echo 2.5.6 ;;
         CC) echo 2.6.8 ;;
          *) echo 2.5.1 ;;
   esac
}
python-version(){
  ## can only cream the version of the python in path in system case.. 
  ## as python-home is being used to set the path 
  case $(python-mode) in 
     system) echo $(python-version-system) ;; 
     source) echo $(python-version-source) ;;
  esac
}


python-mode(){ echo ${PYTHON_MODE:-$(python-mode-default $*)} ; }
python-mode-default(){
  case ${1:-$NODE_TAG} in
          ZZ|C) echo system ;;
     YY|C2|C2R|WW|CC) echo source ;;
             *) echo system ;;
  esac
}



python-cd(){
   cd $(python-site)/$1
}

python-name(){ echo Python-$(python-version) ; }
python-major(){
   local v=$(python-version) 
   echo ${v:0:3} ; 
}
python-sudo(){
   case $(python-mode) in 
     system) echo sudo ;;
     source) echo -n   ;;
          *) echo -n   ;;
   esac
}

python-home(){
   if [ "$(python-mode)" == "source" ]; then
       echo $(local-system-base)/python/$(python-name)
   else
       echo unused ##  too dangerous : /usr  
   fi
}

python-v(){
   case $(python-site) in 
      /opt/local/Library/*) echo "-$(python-major)"  ;;
                         *) echo "" ;;
   esac
}


python-env(){
   local msg="=== $FUNCNAME : "
   local mode=${1:-$(python-mode)}
   #echo $msg mode $mode

   elocal-

   if [ "$mode" == "system" ]; then
      export PYTHON_MODE=system
      #echo $msg mode $mode system branch
      python-unpath $PYTHON_HOME
      export PYTHON_SITE=$(python-site)
      export PYTHON_MODE=system
      #export PYTHONSTARTUP=$ENV_HOME/python/startup.py
   else
      export PYTHON_MODE=source
      #echo $msg mode $mode source branch

      export PYTHON_MAJOR=$(python-major)
      export PYTHON_NAME=$(python-name)
      export PYTHON_HOME=$(python-home)
      export PYTHON_SITE=$(python-home)/lib/python$(python-major)/site-packages
  
     ## THIS IS USED FOR BACKUP PURPOSES ... HENCE CAUTION NEEDED WRT CHANGING THIS 
      export REFERENCE_PYTHON_HOME=$PYTHON_HOME
      #export LD_LIBRARY_PATH=$PYTHON_HOME/lib:$LD_LIBRARY_PATH
      #export PATH=$PYTHON_HOME/bin:$PATH
      python-path $PYTHON_HOME
   fi
   export PYTHON_SUDO=$(python-sudo)
}

python-unpath(){
  local oldhome=$1
  [ -z "$oldhome"  ] && return 0
  [ ! -d "$oldhome" ] && return 0
  env-remove $oldhome/bin
  env-llp-remove $oldhome/lib
}

python-path(){
  local home=$1
  [ -z "$home" ] && return 0 
  [ ! -d "$home" ] && return 0
  env-prepend $home/bin
  env-llp-prepend $home/lib
}

python-ldconfig(){
   env-ldconfig  $PYTHON_HOME/lib
}

python-libdir(){
   case $NODE_TAG in 
      C) echo $PYTHON_HOME/lib ;;
      C2|C2R) echo $PYTHON_HOME/lib ;;
      *) echo ${VIRTUAL_ENV:-ERROR-NO-VIRTUALENV-PYTHON}/lib ;;
   esac
}
python-bindir(){
   echo $PYTHON_HOME/bin
}
python-incdir(){
   echo $PYTHON_HOME/include/python$(python-major)
}

python-site(){
   [ -n "$PYTHON_SITE__" ] && echo $PYTHON_SITE__ && return
   [ -n "$VIRTUAL_ENV" ] && echo $VIRTUAL_ENV/lib/python$(python-major)/site-packages && return 
   python -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())"  
}


python-ln(){
   base-
   base-ln $(python-site) $*
}

python-rln(){
   base-
   base-rln $(python-site) $*
}

python-md5(){ python -c "from md5 import new as _ ; import sys ; print _(''.join(sys.stdin.readlines())).hexdigest() " ; }



python-site-deprecated(){
  case $NODE_TAG in 
    G) echo /Library/Python/$(python-major)/site-packages ;;
    *) python-site- ;;
  esac
}

python-site-(){
    local python=$(which python)
    local archdir=$(dirname $(dirname $python))
    echo $archdir/lib/python$(python-major)/site-packages
}





python-pth(){
  cat $(python-site)/easy-install.pth
}



python-uneasy(){

  ## maybe "easy_install -m  " can do this without the perl ... to investigate


   local eggname=$1
   local pth=$(python-site)/easy-install.pth
   local cmd="$SUDO perl -pi -e \"s/^\.\/$eggname\n$//\" $pth "
   
   echo $msg removing the $eggname entry from $pth
   echo $cmd
   eval $cmd
   
   python-pth
    
}



python-isdevelopdir-(){
   cat $(python-site)/*.egg-link | grep $1 > /dev/null
}




python-unegg(){

   local msg="=== $FUNCNAME :"
   local eggname=$1
   local site=$(python-site)
   
   [ -z "$eggname" ]          && echo $msg ABORT null eggname [$eggname] cannot proceed && return 1 
   [ ${#site} -lt 10 ]        && echo $msg ABORT length of site [$site] is too short... cannot proceed && return 1 
   
   local iwd=$PWD
   cd $site
   [ ! -d $eggname ] && echo $msg ERROR cannot find egg folder $eggname in site $site && return 1
   
   local cmd="$SUDO rm -rf $eggname " 
   echo $msg proceeding with: $cmd
   eval $cmd
   
   cd $iwd
}


python-uninstall(){

  local msg="=== $FUNCNAME :"  
  local eggname=$1
  local answer 
  if [ -z "$PYTHON_UNINSTALL_DIRECTLY" ]; then
     echo $msg delete the egg directory and remove entry from easy-install.pth ... enter YES to proceed
     read answer
  else
     answer=YES
     echo $msg without asking ...  as PYTHON_UNINSTALL_DIRECTLY is defined
  fi
   
  if [ "$answer" == "YES" ]; then
     echo $msg proceeding...
     python-unegg $eggname
     python-uneasy $eggname
  else
     echo $msg skipped
  fi

}











python-x(){  scp $SCM_HOME/python.bash ${1:-$TARGET_TAG}:$SCM_BASE ; }
python-i(){ . $SCM_HOME/python.bash ; }



python-mac-check(){

   find $(python-site) -name '*.so' -exec otool -L {} \; | grep ython

}

#
#   reveals that "libsvn" is hooked up to the wrong python ....
#
# /usr/local/python/Python-2.5.1/lib/python2.5/site-packages/libsvn/_client.so: /System/Library/Frameworks/Python.framework/Versions/2.3/Python (compatibility version 2.3.0, current version 2.3.5)
# /usr/local/python/Python-2.5.1/lib/python2.5/site-packages/libsvn/_core.so:   /System/Library/Frameworks/Python.framework/Versions/2.3/Python (compatibility version 2.3.0, current version 2.3.5)
# /usr/local/python/Python-2.5.1/lib/python2.5/site-packages/libsvn/_delta.so:  /System/Library/Frameworks/Python.framework/Versions/2.3/Python (compatibility version 2.3.0, current version 2.3.5)
# /usr/local/python/Python-2.5.1/lib/python2.5/site-packages/libsvn/_fs.so:     /System/Library/Frameworks/Python.framework/Versions/2.3/Python (compatibility version 2.3.0, current version 2.3.5)
# /usr/local/python/Python-2.5.1/lib/python2.5/site-packages/libsvn/_ra.so:     /System/Library/Frameworks/Python.framework/Versions/2.3/Python (compatibility version 2.3.0, current version 2.3.5)
# /usr/local/python/Python-2.5.1/lib/python2.5/site-packages/libsvn/_repos.so:  /System/Library/Frameworks/Python.framework/Versions/2.3/Python (compatibility version 2.3.0, current version 2.3.5)
# /usr/local/python/Python-2.5.1/lib/python2.5/site-packages/libsvn/_wc.so:     /System/Library/Frameworks/Python.framework/Versions/2.3/Python (compatibility version 2.3.0, current version 2.3.5)
# /usr/local/python/Python-2.5.1/lib/python2.5/site-packages/mod_python/_psp.so:
# /usr/local/python/Python-2.5.1/lib/python2.5/site-packages/neo_cgi.so:
# /usr/local/python/Python-2.5.1/lib/python2.5/site-packages/pysqlite2/_sqlite.so:
#



python-pygments-get(){
	
   pygments_dir=$LOCAL_BASE/python/pygments
   pygments_egg=Pygments-0.7.1-py2.5.egg
   mkdir -p $pygments_dir && cd $pygments_dir
   test -f  $pygments_egg  || curl -o $pygments_egg http://jaist.dl.sourceforge.net/sourceforge/pygments/$pygments_egg
   $SUDO easy_install $pygments_egg 

#Processing Pygments-0.7.1-py2.5.egg
#creating /disk/d4/dayabay/local/python/Python-2.5.1/lib/python2.5/site-packages/Pygments-0.7.1-py2.5.egg
#Extracting Pygments-0.7.1-py2.5.egg to /disk/d4/dayabay/local/python/Python-2.5.1/lib/python2.5/site-packages
#Adding Pygments 0.7.1 to easy-install.pth file
#Installing pygmentize script to /disk/d4/dayabay/local/python/Python-2.5.1/bin
#
#Installed /disk/d4/dayabay/local/python/Python-2.5.1/lib/python2.5/site-packages/Pygments-0.7.1-py2.5.egg
#Processing dependencies for Pygments==0.7.1
#

}




python-crack-egg(){

  path=${1:-dummy.egg}

  [ -f "$path" ] || ( echo the path $path doesnt correspond to a file && return 1 )
  [ -d "$path" ] && ( echo the egg $path is cracked already           && return 1 )

  cd $(dirname $path)
  base=$(basename $path)
  
  sudo mv $base $base.zip
  sudo mkdir $base 
  cd $base
  sudo unzip ../$base.zip
  sudo rm ../$base.zip 
  

}

python-sendmail-test(){

    ## works on H amazingly ... 
    python -c "import smtplib ; s=smtplib.SMTP('localhost'); s.sendmail('me@localhost','blyth@hep1.phys.ntu.edu.tw','test message');  "

}

python-sendmail(){
  
  # 
  #   python-sendmail /path/to/text-message [subject] 
  #
  #        send the text file whose path is passed as a text email ,
  #        if supplied the second argument becomes the subject of the message , 
  #        otherwise the first line of the text file is used as the subject
  #

  local path=${1}
  [ -f "$path" ] || ( echo python-sendmail path $path doesnt exist && return 1 )
  local firstline=$(head -1 $path)
  
  local subject=${2:-$firstline} 
  local to=${3:-$(local-email)}
  local lme="me@localhost"
  local from=${4:-$lme}

   python << EOP
import smtplib
try:
    from email.mime.text import MIMEText
except ImportError:
    from email.MIMEText import MIMEText

# Open a plain text file for reading,  assume ASCII characters only
fp = open( "$path" , 'rb')
# Create a text/plain message
msg = MIMEText(fp.read())
fp.close()

msg['Subject'] = "$subject"
msg['From'] = "$from"
msg['To'] = "$to"

# Send the message via our own SMTP server, but don't include the
# envelope header.
s = smtplib.SMTP()
s.connect()
s.sendmail("$from", "$to", msg.as_string())
#s.sendmail("root@dayabay.ihep.ac.cn","tianxc@ihep.ac.cn", msg.as_string())
s.close()

EOP

}

python-mail(){      cat - | python $(dirname $(python-source))/pipemail.py $* ; }
python-mail-test(){ python-cat | python-mail `local-email` ; }

python-cat(){
  cat << EOC
the subject line
the first message line
the second message line
the last message line
EOC
}





python-sendmail-html(){

  local  me="blyth@hep1.phys.ntu.edu.tw"
  local lme="me@localhost"

  local path=${1}
  [ -f "$path" ] || ( echo python-sendmail path $path doesnt exist && return 1 )
  local firstline=$(head -1 $path)
  
  local subject=${2:-$firstline} 
  local to=${3:-$me}
  local from=${4:-$lme}

  python << EOP

#
# http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/473810
# Send an HTML email with an embedded image and a plain text message for
# email clients that don't want to display the HTML.

import smtplib
from email.MIMEMultipart import MIMEMultipart
from email.MIMEText import MIMEText
from email.MIMEImage import MIMEImage

# Define these once; use them twice!
strFrom = 'from@example.com'
strTo = 'to@example.com'

# Create the root message and fill in the from, to, and subject headers
msg = MIMEMultipart('related')
msg['Subject'] = "$subject"
msg['From'] = "$from"
msg['To'] = "$to"

msg.preamble = 'This is a multi-part message in MIME format.'

# Encapsulate the plain and HTML versions of the message body in an
# 'alternative' part, so message agents can decide which they want to display.
msgAlternative = MIMEMultipart('alternative')
msg.attach(msgAlternative)

msgText = MIMEText('This is the alternative plain text message.')
msgAlternative.attach(msgText)

# We reference the image in the IMG SRC attribute by the ID we give it below
msgText = MIMEText('<b>Some <i>HTML</i> text</b> and an image.<br><img src="cid:image1"><br>Nifty!', 'html')
msgAlternative.attach(msgText)

# This example assumes the image is in the current directory
# fp = open('test.jpg', 'rb')
# msgImage = MIMEImage(fp.read())
# fp.close()
# Define the image's ID as referenced above
# msgImage.add_header('Content-ID', '<image1>')
# msgRoot.attach(msgImage)

# Send the email (this example assumes SMTP authentication is required)

s = smtplib.SMTP()
s.connect()
#s.login('exampleuser', 'examplepass')
s.sendmail("$from", ["$to"], msg.as_string())
s.close()

EOP

}


python-port-on(){
   pkgr-
   pkgr-ln python2.5 python
   pkgr-ln ipython2.5 ipython
}

python-port-off(){
   pkgr-
   pkgr-uln python2.5 python
   pkgr-uln ipython2.5 ipython
}

