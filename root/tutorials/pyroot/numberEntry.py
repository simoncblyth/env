"""

  Example from wlav ...
      http://root.cern.ch/phpBB2/viewtopic.php?t=7822&highlight=tpydispatcher

  Also see  workaround to get  Dispatch(Int_t) to work
     http://root.cern.ch/phpBB2/viewtopic.php?t=6685&highlight=tpydispatcher

     maybe no longer needed as from : TPyDispatcher.h

// pre-defined dispatches, same as per TQObject::Emit(); note that
// Emit() maps exclusively to this set, so several builtin types (e.g.
// Int_t, Bool_t, Float_t, etc.) have been omitted here
   PyObject* Dispatch() { return DispatchVA( 0 ); }
   PyObject* Dispatch( const char* param ) { return DispatchVA( "s", param ); }
   PyObject* Dispatch( Double_t param )    { return DispatchVA( "d", param ); }
   PyObject* Dispatch( Long_t param )      { return DispatchVA( "l", param ); }
   PyObject* Dispatch( Long64_t param )    { return DispatchVA( "L", param ); }



"""

import ROOT 
from ROOT import TGLayoutHints, TGNumberEntry, TPyDispatcher, TGLabel, TGGroupFrame, TGNumberFormat, Form
from ROOT import kFixedWidth, kLHintsTop, kLHintsLeft, kLHintsExpandX, kLHintsBottom, kLHintsRight, kDeepCleanup


class pMyMainFrame( ROOT.TGMainFrame ):
   def __init__( self, parent, width, height ):
      ROOT.TGMainFrame.__init__( self, parent, width, height )

      self.fHor1 = ROOT.TGHorizontalFrame( self, 60, 20, ROOT.kFixedWidth )
      self.fExit = ROOT.TGTextButton( self.fHor1, "&Exit", "gApplication->Terminate(0)" )
      self.fExit.SetCommand( 'TPython::Exec( "raise SystemExit" )' )
      self.fHor1.AddFrame( self.fExit, ROOT.TGLayoutHints( kLHintsTop | kLHintsLeft | kLHintsExpandX, 4, 4, 4, 4 ) )
      self.AddFrame( self.fHor1, TGLayoutHints( kLHintsBottom | kLHintsRight, 2, 2, 5, 1 ) )
   
      self.fNumber = TGNumberEntry( self, 0, 9,999, TGNumberFormat.kNESInteger, TGNumberFormat.kNEANonNegative, TGNumberFormat.kNELLimitMinMax, 0, 99999 )

      self.fLabelDispatch = TPyDispatcher( self.DoSetlabel )

      ## ValueSet(Long_t) is Emitted from TGNumberEntry 
      ## ReturnPressed() is Emitted from TGTextEntry, a base class of TGNumberEntryField 
      self.fNumber.Connect( "ValueSet(Long_t)", "TPyDispatcher", self.fLabelDispatch, "Dispatch()" )
      self.fNumber.GetNumberEntry().Connect( "ReturnPressed()", "TPyDispatcher", self.fLabelDispatch, "Dispatch()" )

      self.AddFrame( self.fNumber, TGLayoutHints( kLHintsTop | kLHintsLeft, 5, 5, 5, 5 ) )
      self.fGframe = TGGroupFrame( self, "Value" )
      self.fLabel = TGLabel( self.fGframe, "No input." )
      self.fGframe.AddFrame( self.fLabel, TGLayoutHints( kLHintsTop | kLHintsLeft, 5, 5, 5, 5) )
      self.AddFrame( self.fGframe, TGLayoutHints( kLHintsExpandX, 2, 2, 1, 1 ) )
   
      self.SetCleanup( kDeepCleanup )
      self.SetWindowName( "Number Entry" )
      self.MapSubwindows()
      self.Resize( self.GetDefaultSize() )
      self.MapWindow()

   def __del__( self ):
      self.Cleanup()

   def DoSetlabel( self ):
      self.fLabel.SetText( Form( "%d" % self.fNumber.GetNumberEntry().GetIntNumber() ) )
      self.fGframe.Layout()


if __name__ == '__main__':
   window = pMyMainFrame( ROOT.gClient.GetRoot(), 50, 50 ) 


