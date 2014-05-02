NuWa DetSim DsOpStackAction
=============================

`DsOpStackAction` 

#. doesnt actually store photons, 
#. just provides classification `fKill/fWaiting/fUrgent` that directs G4 which stack to place things on

   * are the G4 stacks accessible without patching ? **NO** can access stackManager but no access to stacks 
   * convert that into `ChromaPhotonList`

Related
--------

* :doc:`/chroma/chroma_geant4_integration`
* :doc:`/chroma/chroma_nuwa_geant4_integration`
* :doc:`/geant4/event/geant4_stackingaction`
* :doc:`/geant4/event/geant4_stackmanager`



ClassifyNewTrack
-----------------

::

    63 G4ClassificationOfNewTrack DsOpStackAction::ClassifyNewTrack (const G4Track* aTrack) {





DsOpStackAction Basis
----------------------

`NuWa-trunk/dybgaudi/Simulation/DetSim/src/DsOpStackAction.h`::

     26 
     27 class DsOpStackAction :  public GiGaStackActionBase
     28 {   

G4UserStackingAction
-----------------------

* :doc:`/geant4/event/geant4_stackingaction`
* :doc:`/geant4/event/geant4_stackmanager`

`source/event/include/G4UserStackingAction.hh`::

     44 class G4UserStackingAction
     45 {
     46   public:
     47       G4UserStackingAction();
     48       virtual ~G4UserStackingAction();
     49   protected:
     50       G4StackManager * stackManager;
     51   public:
     52       inline void SetStackManager(G4StackManager * value)
     53       { stackManager = value; }
     54 
     55   public: // with description
     56 //---------------------------------------------------------------
     57 // vitual methods to be implemented by user
     58 //---------------------------------------------------------------
     59 //
     60       virtual G4ClassificationOfNewTrack
     61         ClassifyNewTrack(const G4Track* aTrack);


GiGaStackActionBase
--------------------

`NuWa-trunk/lhcb/Sim/GiGa/GiGa/GiGaStackActionBase.h`::

     26 class GiGaStackActionBase:
     27   public virtual IGiGaStackAction   ,
     28   public          GiGaBase
     29 {


ClassifyNewTrack
~~~~~~~~~~~~~~~~~

::

     67   /** From G4:
     68       
     69       Reply G4ClassificationOfNewTrack determined by the
     70       newly coming G4Track.  
     71       
     72       enum G4ClassificationOfNewTrack
     73       {
     74       fUrgent,    // put into the urgent stack
     75       fWaiting,   // put into the waiting stack
     76       fPostpone,  // postpone to the next event
     77       fKill       // kill without stacking
     78       };  
     79       
     80       The parent_ID of the track indicates the origin of it.
     81       G4int parent_ID = aTrack->get_parentID();
     82       
     83       parent_ID = 0 : primary particle
     84       parent_ID > 0 : secondary particle
     85       parent_ID < 0 : postponed from the previous event
     86   */
     87   virtual G4ClassificationOfNewTrack ClassifyNewTrack ( const G4Track* ) ;
     88 


NewStage
~~~~~~~~~~~

::

    089   /** From G4:
    090       
    091       This method is called by G4StackManager when the urgentStack
    092       becomes empty and contents in the waitingStack are transtered
    093       to the urgentStack.
    094       
    095       Note that this method is not called at the begining of each
    096       event, but "PrepareNewEvent" is called.
    097       
    098       In case re-classification of the stacked tracks is needed,
    099       use the following method to request to G4StackManager.
    100       
    101       stackManager->ReClassify();
    102       
    103       All of the stacked tracks in the waitingStack will be re-classified 
    104       by "ClassifyNewTrack" method.
    105       
    106       To abort current event, use the following method.
    107       stackManager->clear();
    108       
    109       Note that this way is valid and safe only for the case it is called
    110       from this user class. The more global way of event abortion is
    111       
    112       G4UImanager * UImanager = G4UImanager::GetUIpointer();
    113       UImanager->ApplyCommand("/event/abort");
    114   */
    115   virtual void                       NewStage         ();


* hmm where is `stackManager` coming from ?





PrepareNewEvent
~~~~~~~~~~~~~~~~~

::

    117   /** From G4:
    118       
    119       This method is called by G4StackManager at the begining of
    120       each event.
    121       
    122       Be careful that the urgentStack and the waitingStack of 
    123       G4StackManager are empty at this moment, because this method
    124       is called before accepting primary particles. Also, note that
    125       the postponeStack of G4StackManager may have some postponed
    126       tracks.
    127   */
    128   virtual void                       PrepareNewEvent  ();




