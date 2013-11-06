
:orphan: True

Understanding PyROOT GUI Threads
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Minimal usage of ROOT/TEve GUI with active ipython shell can be achieved with::

   #!/usr/bin/env python
   import ROOT
   ROOT.PyConfig.GUIThreadScheduleOnce += [ ROOT.TEveManager.Create ]
   if __name__=='__main__':
       ROOT.PyGUIThread.finishSchedule()
       try:
           __IPYTHON__
       except NameError:
           from IPython.Shell import IPShellEmbed
           irgs = ['']
           banner = "entering ipython embedded shell, within the scope of the abtviz instance... try g? for help or the locals() command "
           ipshell = IPShellEmbed(irgs, banner=banner, exit_msg="exiting ipython" )
           ipshell()


But that raises questions 
  * how does this work ? 
  * What threads are involved ? 
  * How would you customize to add a message queue monitoring thread and keep ipython shell operational ?


Answers come from examing PyROOT sources, :file:`root/bindings/pyroot/ROOT.py`

  * `ROOT.PyConfig` is instance of :py:class:`_Configuration` which is implemented in :file:`root/bindings/pyroot/ROOT.py`
  * :py:class:`ROOT.PyGUIThread` isa :py:class:`threading.Thread`


From :file:`ROOT.py`::

   ### helper to prevent GUIs from starving
   def _processRootEvents( controller ):
       import time
       gSystemProcessEvents = _root.gSystem.ProcessEvents
       while controller.keeppolling:
          try:
              gSystemProcessEvents()
              if PyConfig.GUIThreadScheduleOnce:
                  for guicall in PyConfig.GUIThreadScheduleOnce:
                      guicall()
                  PyConfig.GUIThreadScheduleOnce = []
              time.sleep( 0.01 )
          except: # in case gSystem gets destroyed early on exit
              pass

PyROOT **ROOT** in sketch:: 

	class ModuleFacade( types.ModuleType ):
	   def __init__( self, module ):
	      types.ModuleType.__init__( self, 'ROOT' )
	      # lots snipped 

	   def __finalSetup( self ):
	      # lots snipped
	      # root thread, if needed, to prevent GUIs from starving, as needed
	      if self.PyConfig.StartGuiThread and  not ( self.keeppolling or _root.gROOT.IsBatch() ):
		 import threading
		 self.__dict__[ 'keeppolling' ] = 1
		 self.__dict__[ 'PyGUIThread' ] = threading.Thread( None, _processRootEvents, None, ( self, ) )
		 def _finishSchedule( ROOT = self ):

		    #  join([timeout])
		    # Wait until the thread terminates. 
		    # This blocks the calling thread until the thread whose join() method is called terminates ? 
		    # either normally or through an unhandled exception ? or until the optional timeout occurs.
		    #
		    import threading
		    if threading.currentThread() != self.PyGUIThread:
		       while self.PyConfig.GUIThreadScheduleOnce:
			  self.PyGUIThread.join( 0.1 )

		 self.PyGUIThread.finishSchedule = _finishSchedule   # done by conventions ... as first line of the main  
		 self.PyGUIThread.setDaemon( 1 )                     # daemons will not keep python alive, when only daemons are left 
		 self.PyGUIThread.start()



Dumping threads::

	In [18]: for t in threading.enumerate():print t 
	   ....:                                                                                                                                                                                                                                                                         
	<_MainThread(MainThread, started)>
	<Thread(Thread-1, started daemon)>       <<< GUI thread waits on GUI events 

		  
	In [24]: threading.currentThread()
	Out[24]: <_MainThread(MainThread, started)>

	In [26]: threading.activeCount()
	Out[26]: 2



