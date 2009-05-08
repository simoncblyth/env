#!/usr/bin/env python

# Load GaudiPython
from GaudiPython.GaudiAlgs import GaudiAlgo
from GaudiPython import SUCCESS, FAILURE
from GaudiPython import gbl

# Make shortcuts to ROOT classes
TH1F = gbl.TH1F

# Make your algorithm
class ExampleAlg(GaudiAlgo):
    "Example Python Algorithm"
    instance = None
    def __init__(self,name):
        GaudiAlgo.__init__(self,name)
        self.hist = {}
        ExampleAlg.instance = self
        return

    def initialize(self):
        status = GaudiAlgo.initialize(self)
        print "Init ExampleAlg",self.name()
        if status.isFailure(): return status

        # Initialize services
        #self.cableSvc = self.svc('ICableSvc','StaticCableSvc')
        self.histSvc = self.svc('ITHistSvc','THistSvc')
        
        # Make the trigger time histogram
        self.hist["TrigTime"] = TH1F("TrigTime","Trigger Time [s]",
                                     100,0.0,10.0)
        status = self.histSvc.regHist('/file1/examples/trigTime',
                                      self.hist["TrigTime"])
        if status.isFailure(): return status
        # Make the TDC histogram
        self.hist["Tdc"] = TH1F("Tdc","TDC Values",300,0,300)
        status = self.histSvc.regHist('/file1/examples/tdc',
                                      self.hist["Tdc"])


        self.hist["Adc"] = TH1F("Adc","ADC Values", 2000, 0, 2000)
        status = self.histSvc.regHist('/file1/examples/adc',
                                      self.hist["Adc"])


        if status.isFailure(): return status

        return SUCCESS

    def execute(self):
        print "Executing ExampleAlg",self.name()
        evt = self.evtSvc()
        hdr = evt["/Event/Readout/ReadoutHeader"]
        # Exercise 1: Print detector name
        ro = hdr.readout()
        print ro.detector().detName()
        # Exercise 2: Histogram the readout trigger time
        self.hist["TrigTime"].Fill( ro.triggerTime().GetSeconds() )        
        # Exercise 3: Histogram the TDC values
        channelIDs = ro.channels() # Get list of channel IDs in the readout
        counter = 0;
        for channelID in channelIDs:
            channel = ro.channel(channelID) # get the data from the channel
            for tdc in channel.tdc(): # loop over list of TDC values on channel
                self.hist["Tdc"].Fill( tdc )
            for adcClock in channel.adcClocks():
                adcValue = channel.adcByClock(adcClock)
                #print adcClock, adcValue
                counter = counter + 1
                self.hist["Adc"].Fill( adcValue )
        print counter
        return SUCCESS
        
    def finalize(self):        
        print "Finalizing ExampleAlg",self.name()
        status = GaudiAlgo.finalize(self)
        return status

def configure():
    from GaudiSvc.GaudiSvcConf import THistSvc
    histsvc = THistSvc()
    histsvc.Output =["file1 DATAFILE='exampleHist.root' OPT='RECREATE' TYP='ROOT' "]
    return

def run(app):
    '''
    Configure and add an algorithm to job
    '''
    app.ExtSvc += ["StaticCableSvc", "THistSvc"]
    example = ExampleAlg("MyExample")
    app.addAlgorithm(example)
    pass
