




#
#  modpython-apache2-test-run 
#  modpython-apache2-test-prepare
#  modpython-apache2-test-directory
#  modpython-apache2-test-script
#
#
#  modpython-apache2-publisher-prepare
#  modpython-apache2-publisher-directory
#  modpython-apache2-publisher-script
#  modpython-apache2-publisher-open
#
#
#
modpython-test-usage(){ cat << EOU

EOU
}



modpython-apache2-test-run(){
   echo open http://grid1.phys.ntu.edu.tw:6060/test-modpython/hello.py
}

modpython-apache2-test-prepare(){
   handler=hello
   modpython-apache2-test-directory $handler  > $(apache-confdir)/python.conf
   
   mkdir -p $APACHE2_HTDOCS/test-modpython
   modpython-apache2-test-script    > $APACHE2_HTDOCS/test-modpython/$handler.py
   apachectl configtest && apachectl restart
}

modpython-apache2-test-directory(){
    handler=$1
	cat << EOT
#  http://www.modpython.org/live/mod_python-3.3.1/doc-html/inst-testing.html
<Directory $(apache-htdocs)/test-modpython>
	AddHandler mod_python .py
	PythonHandler $handler
	PythonDebug On
</Directory>
EOT
}

modpython-apache2-test-script(){
cat << EOS
from mod_python import apache
def handler(req):
	req.content_type = 'text/plain'
	req.write("Hello World!")
	return apache.OK
EOS
}




modpython-apache2-publisher-prepare(){
   modpython-apache2-publisher-directory  > $APACHE2_HOME/etc/apache2/python.conf
   mkdir -p $APACHE2_HTDOCS/test-modpython
   modpython-apache2-publisher-script    > $APACHE2_HTDOCS/test-modpython/publisher.py
   apachectl configtest && apachectl restart
}

modpython-apache2-publisher-directory(){
	cat << EOT
<Directory $APACHE2_HTDOCS/test-modpython>
	AddHandler mod_python .py
    PythonHandler mod_python.publisher
	PythonDebug On
</Directory>
EOT
}

modpython-apache2-publisher-script(){
# http://www.modpython.org/live/mod_python-3.3.1/doc-html/hand-pub-intro.html
cat << EOS
def say(req, what="NOTHING" ):
	return "I am saying %s" % what
EOS
}

modpython-apache2-publisher-open(){
## NB the peculiar URL
   open http://$HOSTPORT/test-modpython/publisher.py/say?what=hello
}
