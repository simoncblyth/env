
#
#  tracd-run
#  tracd-open
#  trac-initenv-deprecated
#
#
#



export TRACD_HOST=grid1.phys.ntu.edu.tw
export TRACD_PORT=8000


tracd-run(){

  # 
  # the environment LD_LIBRARY_PATH must include $SQLITE_HOME/lib otherwise 
  # cannot access database
  #  ( this is done in sqlite.bash )
  #
  name=${1:-dummy}

  ## switch ownership to self ... probably from www
  sudo chown -R $USER:$USER $SCM_FOLD

  lld
  $PYTHON_HOME/bin/tracd --port $TRACD_PORT $SCM_FOLD/tracs/$name
}

tracd-open(){
   name=${1:-dummy}
   open http://$TRACD_HOST:$TRACD_PORT/$name

## NB no /tracs prefix ... thats an apache-ism
}












trac-initenv-deprecated(){   ## now done in scm-create

  #
  # NB imposed order... the trac envname is forced to match the svn repository name
  #


  name=${1:-dummy}
  reponame="$name"
  repopath="$SVN_PARENT_PATH/$reponame"
  tmplpath=

# initenv <projectname> <db> <repostype> <repospath> <templatepath>

  $PYTHON_HOME/bin/trac-admin $SCM_FOLD/tracs/$name initenv $name sqlite:db/trac.db svn $repopath $tmplpath 

#  sqlite:db/trac.db
#  /tmp/scb/svntest/repo  
#
#  fails...
#
#Creating and Initializing Project
#Failed to create environment. global name 'sqlite' is not defined
#Traceback (most recent call last):
#File "/disk/d4/dayabay/local/python/Python-2.5.1/lib/python2.5/site-packages/trac/scripts/admin.py", line 613, in do_initenv options=options)
#File "/disk/d4/dayabay/local/python/Python-2.5.1/lib/python2.5/site-packages/trac/env.py", line 145, in __init__ self.create(options)
#File "/disk/d4/dayabay/local/python/Python-2.5.1/lib/python2.5/site-packages/trac/env.py", line 250, in createDatabaseManager(self).init_db()
#File "/disk/d4/dayabay/local/python/Python-2.5.1/lib/python2.5/site-packages/trac/db/api.py", line 70, in init_db connector.init_db(**args)
#File "/disk/d4/dayabay/local/python/Python-2.5.1/lib/python2.5/site-packages/trac/db/sqlite_backend.py", line 121, in init_db cnx = sqlite.connect(path, timeout=int(params.get('timeout', 10000)))
#NameError: global name 'sqlite' is not defined
#
#  after setting LD_LIBRARY_PATH to include $SQLITE_HOME/lib and deleting  /tmp/scb/svntest/tracenv 
#  can initenv again successfully ..
#
#Warning:
#You should install the SVN bindings
#
#---------------------------------------------------------------------
#Project environment for 'My Project' created.
#
#You may now configure the environment by editing the file:
#
#  /tmp/scb/svntest/tracenv/conf/trac.ini
#
# If you'd like to take this new project environment for a test drive,
#  try running the Trac standalone web server `tracd`:
#
#    tracd --port 8000 /tmp/scb/svntest/tracenv
#
#	Then point your browser to http://localhost:8000/tracenv.
#   There you can also browse the documentation for your installed
#	version of Trac, including information on further setup (such as
#	deploying Trac to a real web server).
#
#
#   works to some extent but complaint re svn bindings in the web output :
#
#Traceback (most recent call last):
#File "/disk/d4/dayabay/local/python/Python-2.5.1/lib/python2.5/site-packages/trac/web/main.py", line 406, in dispatch_request dispatcher.dispatch(req)
#File "/disk/d4/dayabay/local/python/Python-2.5.1/lib/python2.5/site-packages/trac/web/main.py", line 191, in dispatch chosen_handler = self._pre_process_request(req, chosen_handler)
#File "/disk/d4/dayabay/local/python/Python-2.5.1/lib/python2.5/site-packages/trac/web/main.py", line 263, in _pre_process_request chosen_handler = f.pre_process_request(req, chosen_handler)
#File "/disk/d4/dayabay/local/python/Python-2.5.1/lib/python2.5/site-packages/trac/versioncontrol/api.py", line 73, in pre_process_request self.get_repository(req.authname).sync()
#File "/disk/d4/dayabay/local/python/Python-2.5.1/lib/python2.5/site-packages/trac/versioncontrol/api.py", line 94, in get_repository ((self.repository_type,)*2))
#
#   TracError: Unsupported version control system "svn". Check that the Python bindings for "svn" are correctly installed.
#

}


