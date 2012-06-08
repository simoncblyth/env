# === func-gen- : tools/fabric fgp tools/fabric.bash fgn fabric fgh tools
fabric-src(){      echo tools/fabric.bash ; }
fabric-source(){   echo ${BASH_SOURCE:-$(env-home)/$(fabric-src)} ; }
fabric-vi(){       vi $(fabric-source) ; }
fabric-env(){      elocal- ; }
fabric-usage(){ cat << EOU



  http://docs.fabfile.org/en/1.4.2/index.html
  http://awaseroot.wordpress.com/2012/04/25/fabric-tutorial-2-file-transfer-error-handling/
  pip-2.5 search fabric


  alternatives
  =============

     http://codespeak.net/execnet/index.html
     http://tav.espians.com/fabric-python-with-cleaner-api-and-parallel-deployment-support.html


  idea for usage
  ===============

     fabfile that invokes remote fabfiles that invokes further to 
     test all ssh connections in network of nodes, pulling the results 
     into rst format graphiz node diagram constructed in part on each node
     and collected back to the invoking node

     to stop the recursion getting out of control, need to handle 
     the originating node differently OR use a dated output file that 
     is checked for before proceeding to call buddies


  bizarre : default is not using ssh config
  ==========================================

      just uses /etc/hosts i suppose ?   

      env.use_ssh_config = True

      http://stackoverflow.com/questions/3077281/pythons-fabric-connect-to-a-host-listed-ssh-config



  installs
  ============

    C2 :

        [blyth@cms02 scm]$ which python
	/data/env/system/python/Python-2.5.6/bin/python
	[blyth@cms02 scm]$ which pip
	/data/env/system/python/Python-2.5.6/bin/pip
	[blyth@cms02 scm]$ pip install fabric



    G : 
          macports fabric is old  py25-fabric @0.1.1 (python)  
	  get 1.4.2 via  pip-2.5


          surprised where the fab went 
	       ll  /opt/local/Library/Frameworks/Python.framework/Versions/2.5/bin/fab

          rather than cope with side-effects from PATH changes, 


	  simon:~ blyth$ sudo pip-2.5 install Fabric



simon:~ blyth$ sudo pip-2.5 install Fabric
Password:
Downloading/unpacking Fabric
  Downloading Fabric-1.4.2.tar.gz (182Kb): 182Kb downloaded
  Running setup.py egg_info for package Fabric
    warning: no previously-included files matching '*' found under directory 'docs/_build'
    warning: no files found matching 'fabfile.py'
Downloading/unpacking ssh>=1.7.14 (from Fabric)
  Downloading ssh-1.7.14.tar.gz (794Kb): 794Kb downloaded
  Running setup.py egg_info for package ssh
Downloading/unpacking pycrypto>=2.1,!=2.4 (from ssh>=1.7.14->Fabric)
  Downloading pycrypto-2.6.tar.gz (443Kb): 443Kb downloaded
  Running setup.py egg_info for package pycrypto
Installing collected packages: Fabric, ssh, pycrypto
  Running setup.py install for Fabric
    warning: no previously-included files matching '*' found under directory 'docs/_build'
    warning: no files found matching 'fabfile.py'
    Installing fab script to /opt/local/Library/Frameworks/Python.framework/Versions/2.5/bin
  Running setup.py install for ssh
  Running setup.py install for pycrypto
    checking for gcc... gcc








EOU
}
fabric-dir(){ echo $(local-base)/env/tools/tools-fabric ; }
fabric-cd(){  cd $(fabric-dir); }
fabric-mate(){ mate $(fabric-dir) ; }
fabric-get(){
   local dir=$(dirname $(fabric-dir)) &&  mkdir -p $dir && cd $dir

}

fabric-osx-ln(){
   cd /opt/local/bin
   sudo ln -s  /opt/local/Library/Frameworks/Python.framework/Versions/2.5/bin/fab fab
}

