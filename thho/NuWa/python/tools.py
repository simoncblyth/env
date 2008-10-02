from GaudiPython.GaudiAlgs import GaudiAlgo
from GaudiPython import *
import ROOT
import PyCintex
from array import array

PyCintex.loadDict("libConventionsDict")

def nint(value):
    "Return the nearest integer"
    return int(round(value))

gTriggers = []
gCrates = []
gReadouts = []

class SimpleTriggerAlg(GaudiAlgo):
    "Primitive trigger simulation"
    def __init__(self,name):
        GaudiAlgo.__init__(self,name)
        print "Making SimpleTriggerAlg",name
        
        global gCrates
        global gTriggers

        self.m_threshold = 30
        # don't retrigger until after 40 clock cycles at 80 MHz (= 500 ns)
        self.m_retriggerCycles = 40
        self.m_crates = gCrates
        self.m_triggers = gTriggers

    def initialize(self):
        status = GaudiAlgo.initialize(self)
        return status

    def finalize(self):
        status = GaudiAlgo.finalize(self)
        return status

    def execute(self):
        "Get event service"
        app = AppMgr()
        evt = app.evtSvc()
        
        "Fill trigger data according to crate signals"
        # only consider AD detectors
        detectors = [ROOT.DayaBay.Detector(0x01,1),ROOT.DayaBay.Detector(0x01,2),
                     ROOT.DayaBay.Detector(0x02,1),ROOT.DayaBay.Detector(0x02,2),
                     ROOT.DayaBay.Detector(0x04,1),ROOT.DayaBay.Detector(0x04,2),
                     ROOT.DayaBay.Detector(0x04,3),ROOT.DayaBay.Detector(0x04,4)]
        global gCrates
        global gTriggers
        gCrates = []
        gTriggers = []
        self.m_crates = gCrates
        self.m_triggers = gTriggers
        for detector in detectors:
            elecHdr = evt['/Event/Elec/ElecHeader']
            crate_map = elecHdr.crateHeader().crates()
            crate = crate_map[detector]
            if crate == None:
                print "SimpleTriggerAlg.mutate: Crate not found for detector ",detector
                continue
            self.m_crates.append(crate)
        # FIXME: currently only does local multiplicity trigger
        boards = range(1,13)   # 12 PMT boards
        crateIdx = 0           # input for trigger to know what crate to associate with
        for crate in self.m_crates:
            # FIXME: Trigger on first twelve FEE boards is hard-coded
            triggerNhit = []
            site = crate.detector().site()
            det  = crate.detector().detectorId()
            for board in range(len(boards)):
                hits = crate.nhitSignal(ROOT.DayaBay.FeeChannelId(board,1,site,det).boardId())
                if len(hits)==0:
                    continue
                if (len(hits) != 0) and (len(triggerNhit) == 0):
                    triggerNhit = [x*0 for x in range(len(hits))]
                for hit in range(len(hits)):
                    triggerNhit[hit] = triggerNhit[hit] + hits[hit]
            isReset = True
            resetCycle = 0
            for i in range(len(triggerNhit)):
                aboveThreshold = triggerNhit[i] >= self.m_threshold
                if not isReset and i == resetCycle:
                    # Reset Trigger
                    isReset = True
                if aboveThreshold and isReset:
                    triggerCommand = TSTriggerCommand()
                    triggerCommand.m_clockCycle = i
                    triggerCommand.m_type = 0
                    triggerCommand.m_crateIdx = crateIdx
                    self.m_triggers.append(triggerCommand)
                    isReset = False
                    resetCycle = i + self.m_retriggerCycles
            crateIdx += 1
        return SUCCESS

class TSTriggerCommand:
    def __init__(self):
        self.m_type = 0
        self.m_clockCycle = 0
        self.m_crateIdx = 0

class SimpleFPGAAlg(GaudiAlgo):
    "Simple FPGA analysis and readout"
    def __init__(self,name):
        GaudiAlgo.__init__(self,name)
        print "Making SimpleFPGAAlg",name

        global gReadouts
        
        self.m_baseFreq = 40e6
        self.m_nhitCycles = 2
        self.m_tdcCycles = 16
        self.m_adcCycles = 1
        # Readout window start in seconds relative to trigger time
        self.m_readoutStart = -100.e-9
        self.m_readoutStop = 400.e-9
        self.m_lowGainThreshold = 4000

        self.m_readouts = gReadouts

    def initialize(self):
        status = GaudiAlgo.initialize(self)
        return status

    def finalize(self):
        status = GaudiAlgo.finalize(self)
        return status

    def execute(self):
        "Define crates and triggers from SimpleTriggerAlg"
        global gCrates
        global gTriggers
        global gReadouts
        gReadouts = []
        crates = gCrates
        triggers = gTriggers
        self.m_readouts = gReadouts

        nTriggers = len(triggers)
        for i in range(nTriggers):
            triggerCommand = triggers[i]
            crateIdx = triggerCommand.m_crateIdx
            self.processOneReadout(crates[crateIdx], triggerCommand)
        return SUCCESS

    def processOneReadout(self, crate, triggerCommand): 
        "Append readout according to the crate and trigger command"
        nhitFreq = self.m_baseFreq * self.m_nhitCycles
        adcFreq  = self.m_baseFreq * self.m_adcCycles
        crateReadoutNhitOffset = nint(triggerCommand.m_clockCycle
                                      + (self.m_readoutStart * nhitFreq)) # offset wrt sim start
        if crateReadoutNhitOffset < 0:
            crateReadoutNhitOffset = 0
        crateReadoutOffsetTime = crateReadoutNhitOffset / float(nhitFreq) # offset wrt sim start
        readout = FeeCrateReadout()
        readout.m_site = crate.detector().site()
        readout.m_detector = crate.detector().detectorId()
        readout.m_startTime = crate.header().header().earliest().GetSeconds() + crateReadoutOffsetTime # absolute time
        readout.m_triggerClock = (triggerCommand.m_clockCycle
                                  - crateReadoutNhitOffset)  # clock wrt readout start
        readout.m_triggerType = triggerCommand.m_type
        readout.m_readoutCycles= nint((self.m_readoutStop -self.m_readoutStart)
                                      * adcFreq)

        "Loop through all channels"
        # loop through channels depends on having only ADs as possible detectors (no RPC,IWS,OWS)
        channelMap = crate.channelData()
        boards = range(1,13)     # 12 AD PMT boards
        connectors = range(1,17) # 16 AD PMT connectors per AD PMT board
        for board in boards:
            for connector in connectors:
                channelID = ROOT.DayaBay.FeeChannelId(board,connector,crate.detector().site(),crate.detector().detectorId())
                channel = channelMap[channelID]
                if len(channel.tdc()) < 1:
                    #print "No TDCs on board",board,"connector",connector
                    continue
                #print "Processing:",board, connector
                self.processChannel(channelID, crateReadoutNhitOffset,channel,readout)
        self.m_readouts.append(readout)

    def processChannel(self, channelID, crateReadoutNhitOffset, channelInput, readout):
        "Gather readout data for a single channel"
        channelReadout = None
        nTDC = len(channelInput.tdc())
        tdcFreq = self.m_tdcCycles*self.m_baseFreq
        for i in range(nTDC):
            tdc = channelInput.tdc()
            tdcValue = tdc[i]
            readoutTDC = ( tdcValue
                           - nint((crateReadoutNhitOffset * self.m_tdcCycles)
                                  / float(self.m_nhitCycles))) 
            dT = readoutTDC / float(tdcFreq)
            if dT >= 0 and dT < self.m_readoutStop - self.m_readoutStart:
                channelReadout = FeeChannelReadout()
                channelReadout.m_channelID = channelID
                channelReadout.addTDC(readoutTDC)
        if not channelReadout:
            print "Cannot process channel",channelID
            return

        nADC = len(channelInput.adcHigh())
        adcLowG = []
        adcHighG = []
        adcLowGMax = 0
        adcHighGMax = 0
        adcLowGMaxIndex = 0
        adcHighGMaxIndex = 0
        crateReadoutADCOffset = nint( crateReadoutNhitOffset
                                      * float(self.m_adcCycles)
                                      / float(self.m_nhitCycles) )
        minCycle = crateReadoutADCOffset
        maxCycle = minCycle + readout.m_readoutCycles
        if minCycle < 0:
            print "SimpleFPGAAlg.processChannel: invalid min cycle",minCycle
            minCycle = 0
        if maxCycle > nADC:
            print "SimpleFPGAAlg.processChannel: invalid max cycle",maxCycle
            maxCycle = nADC
        # Cycle over ADC samples in readout window, and find peak
        for i in range(maxCycle-minCycle):
            adcH = channelInput.adcHigh()[i+minCycle]
            adcL = channelInput.adcLow()[i+minCycle]
            adcHighG.append(adcH)
            adcLowG.append(adcL)
            if(adcHighG[i]>adcHighGMax):
               adcHighGMax = adcHighG[i]
               adcHighGMaxIndex = i
            if(adcLowG[i]>adcLowGMax):
               adcLowGMax = adcLowG[i]
               adcLowGMaxIndex = i
        if(adcHighGMax >= self.m_lowGainThreshold):
            channelReadout.m_adcGain = 2 # FIXME: enums not being generated in dictionary
            channelReadout.addADC(adcLowGMax,adcLowGMaxIndex) # only put maximums for adc info
            adc = adcLowG
        else:
            channelReadout.m_adcGain = 1
            channelReadout.addADC(adcHighGMax,adcHighGMaxIndex)
            adc = adcHighG

        # Simple Baseline Analysis
        channelReadout.m_baseline = adc[0]

        readout.addChannel(channelReadout)

class FeeCrateReadout:
    def __init__(self):
        self.m_site = 0          # detector site
        self.m_detector = 0      # detector ID
        self.m_startTime = 0     # readout start time, absolute
        self.m_triggerClock = 0  # trigger clock cycle, wrt readout start (clock defined by nhitFreq)
        self.m_triggerType = 0   # type of trigger (0=unknown, 1=crossing, 2=above)
        self.m_readoutCycles = 0 # number of readout clock cycles (defined by adcFreq)
        self.m_channelReadout = []
    def addChannel(self,channelReadout):
        self.m_channelReadout += [channelReadout]

class FeeChannelReadout:
    def __init__(self):
        self.m_channelID = 0     # channel ID
        self.m_nHit = 0          # number of hits on this channel
        self.m_tdcValues = []    # tdc clock values
        self.m_adcValues = []    # max ADC in readout window
        self.m_adcGain = 0       # ADC gain for adcMax (0=unknown, 1=high, 2=low)
        self.m_adcClock = []     # adc clock, wrt readout start
        self.m_baseline = 0
    def addTDC(self,value):
        self.m_tdcValues += [value]
    def addADC(self,value,clock):
        self.m_adcValues += [value]
        self.m_adcClock += [clock]

class FeeReadoutWriter(GaudiAlgo):
    "Simple FEE readout data writer"
    def __init__(self,name):
        GaudiAlgo.__init__(self,name)
        print "Making FeeReadoutWriter",name
        
        self.m_outputFilename = ""
        self.m_outFile = None

    def setOutputFile(self,filename):
        self.m_outputFilename = filename

    def initialize(self):
        "Open and prepare output file for raw data"
        self.m_outFile = ROOT.TFile(self.m_outputFilename,"RECREATE")
        if self.m_outFile==None:
            print "FeeReadoutWriter.initialize: Failed to open output file=",self.m_outputFilename
        self.m_maxTDC = 4096
        self.m_maxADC = 4096
        self.m_maxNHit = 256
        
        self.m_site = array('i',[0])
        self.m_detector = array('i',[0])
        self.m_startTime = array('d',[0.])
        self.m_triggerClock = array('i',[0])
        self.m_triggerType = array('i',[0])
        self.m_nTDC = array('i',[0])
        self.m_tdc = array('i',self.m_maxTDC*[0])
        self.m_tdcChannel = array('i',self.m_maxTDC*[0])
        self.m_nADC = array('i',[0])
        self.m_adc = array('i',self.m_maxADC*[0])
        self.m_adcChannel = array('i',self.m_maxADC*[0])
        self.m_adcGain = array('d',self.m_maxADC*[0])
        self.m_adcClock = array('i',self.m_maxADC*[0])
        self.m_adcBaseline = array('i',self.m_maxADC*[0])
        
        self.m_tree = ROOT.TTree("readoutDataTree","Readout Data Output")
        self.m_tree.Branch("site",self.m_site,"site/I")
        self.m_tree.Branch("detector",self.m_detector,"detector/I")
        self.m_tree.Branch("readoutTime",self.m_startTime,"readoutTime/D")
        self.m_tree.Branch("triggerClock",self.m_triggerClock,"triggerClock/I")
        self.m_tree.Branch("triggerType",self.m_triggerType,"triggerType/I")        
        self.m_tree.Branch("nTDC",self.m_nTDC,"nTDC/I")
        self.m_tree.Branch("tdc",self.m_tdc,"tdc[nTDC]/I")
        self.m_tree.Branch("tdcChannel",self.m_tdcChannel,"tdcChannel[nTDC]/I")
        self.m_tree.Branch("nADC",self.m_nADC,"nADC/I")
        self.m_tree.Branch("adc",self.m_adc,"adc[nADC]/I")        
        self.m_tree.Branch("adcChannel",self.m_adcChannel,"adcChannel[nADC]/I")
        self.m_tree.Branch("adcGain",self.m_adcGain,"adcGain[nADC]/D")
        self.m_tree.Branch("adcClock",self.m_adcClock,"adcClock[nADC]/I")
        self.m_tree.Branch("adcBaseline",self.m_adcBaseline,"adcBaseline[nADC]/I")
        
        status = GaudiAlgo.initialize(self)
        return status
    
    def finalize(self):
        "Close output file"
        self.m_outFile.cd()
        self.m_tree.Write()
        self.m_outFile.Close()
        self.m_outFile = None

        status = GaudiAlgo.finalize(self)
        return status

    def write(self,readouts):
        "Write out the readouts to a file"
        nReadouts = len(readouts)
        print "  writing",nReadouts,"readouts to",self.m_outputFilename,"at entry",(self.m_tree.GetEntries()+1)
        for i in range(nReadouts):
            # Process each readout
            readout = readouts[i]
            self.m_site[0] = readout.m_site
            self.m_detector[0] = readout.m_detector
            self.m_startTime[0] = readout.m_startTime
            self.m_triggerClock[0] = readout.m_triggerClock
            self.m_triggerType[0] = readout.m_triggerType
            nTDC = 0
            nADC = 0
            for j in range(len(readout.m_channelReadout)):
                channelReadout = readout.m_channelReadout[j]
                for tdcValue in channelReadout.m_tdcValues:
                    self.m_tdc[nTDC] = tdcValue
                    self.m_tdcChannel[nTDC] = channelReadout.m_channelID.fullPackedData()
                    self.m_tdc
                    nTDC += 1
                for adcIndex in range(len(channelReadout.m_adcValues)):
                    self.m_adc[nADC] = channelReadout.m_adcValues[adcIndex]
                    self.m_adcChannel[nADC] = channelReadout.m_channelID.fullPackedData()
                    self.m_adcGain[nADC] = channelReadout.m_adcGain
                    self.m_adcClock[nADC] = channelReadout.m_adcClock[adcIndex]
                    self.m_adcBaseline[nADC] = channelReadout.m_baseline
                    nADC += 1
            self.m_nTDC[0] = nTDC
            self.m_nADC[0] = nADC
            self.m_tree.Fill()
        return SUCCESS

    def execute(self):
        "Write readouts to a file"
        global gReadouts
        readouts = gReadouts

        return self.write(readouts)
