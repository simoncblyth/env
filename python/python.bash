
python-src(){   echo python/python.bash ; }
python-source(){ echo ${BASH_SOURCE:-$ENV_HOME/$(python-src)} ; }
python-vi(){     vi $(python-source) ; }
python-url(){    echo $(env-url)/$python-src ; }
python-usage(){

cat << EOU


    PYTHON_SITE : $PYTHON_SITE

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
         
    python-cd     
       cd $(python-site)  


    python-mail 
           echo hello testing python-mail  | python-mail `local-email` 

    python-mail-test

           python-mail-test
           >  Attempting to send email to recipient:[blyth@hep1.phys.ntu.edu.tw] from:[me@localhost] message lines:[4] 

        NB for the mailing to work, a mailer needs to be running, eg on G :
            sudo postfix start
            > postfix/postfix-script: starting the Postfix mail system



EOU




}


pexpect-(){      . $ENV_HOME/python/pexpect.bash   && pexpect-env   $* ; } 
configobj-(){    . $ENV_HOME/python/configobj.bash && configobj-env $* ; }
pythonbuild-(){  . $ENV_HOME/python/pythonbuild/pythonbuild.bash && pythonbuild-env $* ; } 


python-ls(){

   ls -l $(python-site)/

}


python-cd(){
   cd $(python-site)
}

python-mode(){ echo ${PYTHON_MODE:-source} ; }
python-name(){ echo Python-2.5.1 ; }
python-major(){ echo 2.5 ; }
python-home(){
   if [ "$(python-mode)" == "source" ]; then
       echo $(local-system-base)/python/$(python-name)
   else
       echo unused-?
   fi
}

python-sudo(){
   case $(python-mode) in 
     system) echo sudo ;;
     source) echo -n   ;;
          *) echo -n   ;;
   esac
}


python-env(){

   local mode=${1:-$(python-mode)}
   elocal-

   export PYTHON_SUDO=$(python-sudo)

   if [ "$mode" == "system" ]; then
      export PYTHON_SITE=$(python-site)
      #export PYTHONSTARTUP=$ENV_HOME/python/startup.py
   else
      export PYTHON_MAJOR=$(python-major)
      export PYTHON_NAME=$(python-name)
      export PYTHON_HOME=$(python-home)
      export PYTHON_SITE=$(python-home)/lib/python$(python-major)/site-packages
      
      python-path
  
     ## THIS IS USED FOR BACKUP PURPOSES ... HENCE CAUTION NEEDED WRT CHANGING THIS 
      export REFERENCE_PYTHON_HOME=$PYTHON_HOME
   fi
}


python-site(){
  python -c "from distutils.sysconfig import get_python_lib; print get_python_lib()"
}


python-ln(){
    local msg="=== $FUNCNAME :";
    local path=$1
    [ ! -d "$path" ] && echo $msg ABORT no such path $path && return 1
    local lnk=$(python-site)/$(basename $path);
    [ -L "$lnk" ] && echo $msg link $lnk is already present ... skipping && return 0;
    local cmd="sudo ln -s $path $(python-site)/$(basename $path)";
    echo $msg $cmd;
    eval $cmd
}


python-site-deprecated(){
  case $NODE_TAG in 
    G) echo /Library/Python/2.5/site-packages ;;
    *) python-site- ;;
  esac
}

python-site-(){
    local python=$(which python)
    local archdir=$(dirname $(dirname $python))
    echo $archdir/lib/python2.5/site-packages
}

python-path(){

  [ -z $PYTHON_HOME ] && echo $msg skip as no PYTHON_HOME && return 1  

  env-prepend $PYTHON_HOME/bin
  env-llp-prepend $PYTHON_HOME/lib
  

}

python-ldconfig(){

   env-ldconfig  $PYTHON_HOME/lib


}



python-libdir(){
   echo $PYTHON_HOME/lib
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
from email.mime.text import MIMEText

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

