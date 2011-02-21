#!/usr/bin/env python
"""
   Exploring how PyROOT-GUI-Eve threading works ... 
      * in order to add a pika/rabbitmq queue monitor at python level

"""
import ROOT
ROOT.PyConfig.GUIThreadScheduleOnce += [ ROOT.TEveManager.Create ]

"""
   ROOT.PyConfig is instance of _Configuration which is implemenrted in root/bindings/pyroot/ROOT.py

   In [8]: ROOT.PyConfig.                                                                                                                                                                                                                                                           
ROOT.PyConfig.GUIThreadScheduleOnce     ROOT.PyConfig._Configuration__setGTS    ROOT.PyConfig.__getattribute__          ROOT.PyConfig.__new__                   ROOT.PyConfig.__setattr__               
ROOT.PyConfig.IgnoreCommandLineOptions  ROOT.PyConfig.__class__                 ROOT.PyConfig.__hash__                  ROOT.PyConfig.__reduce__                ROOT.PyConfig.__slots__                 
ROOT.PyConfig.StartGuiThread            ROOT.PyConfig.__delattr__               ROOT.PyConfig.__init__                  ROOT.PyConfig.__reduce_ex__             ROOT.PyConfig.__str__                   
ROOT.PyConfig._Configuration__getGTS    ROOT.PyConfig.__doc__                   ROOT.PyConfig.__module__                ROOT.PyConfig.__repr__                  ROOT.PyConfig._gts                      



In [10]: ROOT.PyGUIThread.__class__                                                                                                                                                                                                                                              
Out[10]: <class 'threading.Thread'>

In [11]: ROOT.PyGUIThread.                                                                                                                                                                                                                                                       
ROOT.PyGUIThread._Thread__args         ROOT.PyGUIThread._Thread__kwargs       ROOT.PyGUIThread._Verbose__verbose     ROOT.PyGUIThread.__init__              ROOT.PyGUIThread.__str__               ROOT.PyGUIThread.isDaemon
ROOT.PyGUIThread._Thread__block        ROOT.PyGUIThread._Thread__name         ROOT.PyGUIThread.__class__             ROOT.PyGUIThread.__module__            ROOT.PyGUIThread.__weakref__           ROOT.PyGUIThread.join
ROOT.PyGUIThread._Thread__bootstrap    ROOT.PyGUIThread._Thread__started      ROOT.PyGUIThread.__delattr__           ROOT.PyGUIThread.__new__               ROOT.PyGUIThread._note                 ROOT.PyGUIThread.run
ROOT.PyGUIThread._Thread__daemonic     ROOT.PyGUIThread._Thread__stderr       ROOT.PyGUIThread.__dict__              ROOT.PyGUIThread.__reduce__            ROOT.PyGUIThread._set_daemon           ROOT.PyGUIThread.setDaemon
ROOT.PyGUIThread._Thread__delete       ROOT.PyGUIThread._Thread__stop         ROOT.PyGUIThread.__doc__               ROOT.PyGUIThread.__reduce_ex__         ROOT.PyGUIThread.finishSchedule        ROOT.PyGUIThread.setName
ROOT.PyGUIThread._Thread__exc_info     ROOT.PyGUIThread._Thread__stopped      ROOT.PyGUIThread.__getattribute__      ROOT.PyGUIThread.__repr__              ROOT.PyGUIThread.getName               ROOT.PyGUIThread.start
ROOT.PyGUIThread._Thread__initialized  ROOT.PyGUIThread._Thread__target       ROOT.PyGUIThread.__hash__              ROOT.PyGUIThread.__setattr__           ROOT.PyGUIThread.isAlive      


From ROOT.py ....

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




Python threading background

   *  http://docs.python.org/library/threading.html#thread-objects

     threading.Thread(self, group=None, target=None, name=None, args=(), kwargs={}, verbose=None)


In [18]: for t in threading.enumerate():print t 
   ....:                                                                                                                                                                                                                                                                         
<_MainThread(MainThread, started)>
<Thread(Thread-1, started daemon)>       <<< GUI thread waits on GUI events 

          
In [24]: threading.currentThread()
Out[24]: <_MainThread(MainThread, started)>

In [26]: threading.activeCount()
Out[26]: 2

"""

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


