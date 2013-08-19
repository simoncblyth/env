#!/usr/bin/env python
"""
GOOGLEPERFTOOLS CPU PROFILE TEXT ANALYSIS
============================================

Meaning the columns

1. `nsmp` Number of profiling samples in this function
2. `psmp` Percentage of profiling samples in this function
3. `fsmp` Percentage of profiling samples in the functions printed so far
4. `nsub` Number of profiling samples in this function and its callees
5. `psub` Percentage of profiling samples in this function and its callees
6. `name` Function name


Usage::

    [blyth@belle7 gperftools]$  gproftext.py  /tmp/nuwa.perf.txt --filter 'psub > 5 and nsmp > 0'  --sort psub --reverse
    Total: 51006 samples

    filter : psub > 5 and nsmp > 0      sort : psub    reverse : True

      nsmp       psmp %       fsmp             nsub       psub %         name 
      42.0        0.1 %       79.3             47777.0    93.7 %          _init@80579cc 
      3.0         0.0 %       98.4             42679.0    83.7 %          PyEval_EvalFrameEx 
      68.0        0.1 %       70.3             26885.0    52.7 %          G4TrackingManager::ProcessOneTrack 
      16.0        0.0 %       91.4             26902.0    52.7 %          G4EventManager::DoProcessing 
      174.0       0.3 %       47.8             26035.0    51.0 %          G4SteppingManager::Stepping 
      110.0       0.2 %       58.7             12153.0    23.8 %          G4EventManager::StackTracks 
      99.0        0.2 %       61.3             11895.0    23.3 %          G4StackManager::PushOneTrack 
      687.0       1.3 %       21.1             11143.0    21.8 %          DsFastMuonStackAction::ClassifyNewTrack 
      50.0        0.1 %       76.6             9303.0     18.2 %          G4SteppingManager::InvokePostStepDoItProcs 
      186.0       0.4 %       45.3             9145.0     17.9 %          G4SteppingManager::InvokePSDIP 
      1.0         0.0 %       99.2             8316.0     16.3 %          DataSvc::retrieveEntry 
      1.0         0.0 %       99.2             8316.0     16.3 %          DataSvc::retrieveObject@938080 
      4.0         0.0 %       97.5             8189.0     16.1 %          .L2198 
      2.0         0.0 %       98.6             8200.0     16.1 %          DataSvc::loadObject@938dfc 
      12.0        0.0 %       93.8             8183.0     16.0 %          XmlGenericCnv::createObj 
      1.0         0.0 %       99.9             7453.0     14.6 %          xercesc_2_8::DOMDeepNodeListImpl::item 
      182.0       0.4 %       46.4             7410.0     14.5 %          xercesc_2_8::DOMDeepNodeListImpl::cacheItem 
      4924.0      9.7 %       9.7              6471.0     12.7 %          xercesc_2_8::DOMDeepNodeListImpl::nextMatchingElementAfter 
      140.0       0.3 %       52.0             6276.0     12.3 %          G4SteppingManager::DefinePhysicalStepLength 
      98.0        0.2 %       61.7             5946.0     11.7 %          GiGaStepActionSequence::UserSteppingAction 
      1.0         0.0 %       99.2             5543.0     10.9 %          DetectorElement::childIDetectorElements 
      348.0       0.7 %       32.1             5516.0     10.8 %          GetTouchableName 
      8.0         0.0 %       95.8             4743.0      9.3 %          G4VProcess::AlongStepGPIL 
      370.0       0.7 %       30.7             4634.0      9.1 %          UnObserverStepAction::UserSteppingAction 
      117.0       0.2 %       56.5             4590.0      9.0 %          G4Transportation::AlongStepGetPhysicalInteractionLength 
      138.0       0.3 %       52.3             4378.0      8.6 %          G4Navigator::ComputeStep 
      1701.0      3.3 %       13.0             4092.0      8.0 %          std::vector::operator[] 
      8.0         0.0 %       95.8             3565.0      7.0 %          TH2DE::GetBestDetectorElement 
      137.0       0.3 %       52.5             3471.0      6.8 %          G4VoxelNavigation::ComputeStep 
      111.0       0.2 %       58.5             3250.0      6.4 %          std::ostream::operator<< 
      89.0        0.2 %       62.2             3120.0      6.1 %          RuleParser::AndRule::select 
      57.0        0.1 %       74.9             3101.0      6.1 %          std::num_put::do_put 
      163.0       0.3 %       49.1             3054.0      6.0 %          RuleParser::EQ_Rule::select 
      36.0        0.1 %       81.3             2845.0      5.6 %          CLHEP::operator<< 
      277.0       0.5 %       36.9             2588.0      5.1 %          DsG4OpBoundaryProcess::PostStepDoIt 


    [blyth@belle7 20130816-1754]$ gproftext.py  /tmp/nuwa.perf.txt --pattern "G4.*"
    Total: 51006 samples

    filter : psub > 10 and nsmp > 0      sort : psub    reverse : False

      nsmp       psmp %       fsmp             nsub       psub %         name 
      140.0       0.3 %       52.0             6276.0     12.3 %          G4SteppingManager::DefinePhysicalStepLength 
      186.0       0.4 %       45.3             9145.0     17.9 %          G4SteppingManager::InvokePSDIP 
      50.0        0.1 %       76.6             9303.0     18.2 %          G4SteppingManager::InvokePostStepDoItProcs 
      99.0        0.2 %       61.3             11895.0    23.3 %          G4StackManager::PushOneTrack 
      110.0       0.2 %       58.7             12153.0    23.8 %          G4EventManager::StackTracks 
      174.0       0.3 %       47.8             26035.0    51.0 %          G4SteppingManager::Stepping 
      68.0        0.1 %       70.3             26885.0    52.7 %          G4TrackingManager::ProcessOneTrack 
      16.0        0.0 %       91.4             26902.0    52.7 %          G4EventManager::DoProcessing 


    [blyth@belle7 20130816-1754]$ gproftext.py  /tmp/nuwa.perf.txt --pattern "PostStepDoIt" --filter 1==1 
    Total: 51006 samples

    filter : 1==1      sort : psub    reverse : False

      nsmp       psmp %       fsmp             nsub       psub %         name 
      7.0         0.0 %       96.2             10.0        0.0 %          G4VRestDiscreteProcess::PostStepDoIt 
      3.0         0.0 %       98.3             8.0         0.0 %          G4VDiscreteProcess::PostStepDoIt 
      2.0         0.0 %       98.7             25.0        0.0 %          G4LowEnergyRayleigh::PostStepDoIt 
      1.0         0.0 %       99.3             3.0         0.0 %          DsG4OpRayleigh::PostStepDoIt 
      1.0         0.0 %       99.4             20.0        0.0 %          G4OpAbsorption::PostStepDoIt 
      0.0         0.0 %       100.0            12.0        0.0 %          G4LowEnergyBremsstrahlung::PostStepDoIt 
      0.0         0.0 %       100.0            1.0         0.0 %          G4LowEnergyGammaConversion::PostStepDoIt 
      0.0         0.0 %       100.0            9.0         0.0 %          G4LowEnergyPhotoElectric::PostStepDoIt 
      0.0         0.0 %       100.0            16.0        0.0 %          G4VEnergyLossProcess::PostStepDoIt 
      2.0         0.0 %       98.7             37.0        0.1 %          G4LowEnergyCompton::PostStepDoIt 
      2.0         0.0 %       98.8             72.0        0.1 %          G4VMultipleScattering::PostStepDoIt 
      1.0         0.0 %       99.3             38.0        0.1 %          G4LowEnergyIonisation::PostStepDoIt 
      81.0        0.2 %       65.6             719.0       1.4 %          DsG4Cerenkov::PostStepDoIt 
      87.0        0.2 %       63.4             1663.0      3.3 %          G4Transportation::PostStepDoIt 
      393.0       0.8 %       28.4             2332.0      4.6 %          DsG4Scintillation::PostStepDoIt 
      277.0       0.5 %       36.9             2588.0      5.1 %          DsG4OpBoundaryProcess::PostStepDoIt 
      50.0        0.1 %       76.6             9303.0     18.2 %          G4SteppingManager::InvokePostStepDoItProcs 



Parse and sort the text columns from googleperftools created with commands like::

    [blyth@belle7 20130816-1754]$ pprof --text $(which python) /tmp/nuwa.perf > /tmp/nuwa.perf.txt
    Using local file /data1/env/local/dyb/external/Python/2.7/i686-slc5-gcc41-dbg/bin/python.
    Using local file /tmp/nuwa.perf.
    Removing _init from all stack traces. 

For a top-down "container" view filter as below, the nsmp > 0 avoids uninteresting "global" containers

* psub > 10 and nsmp > 0     





"""
import re
from optparse import OptionParser

def interp_nf(txt,n):     
    elem = txt.split()
    assert len(elem) == n, txt
    return map(float,elem)
 
class Line(dict):
    fmt = "  %(nsmp)-8s %(psmp)6s %%       %(fsmp)-10s       %(nsub)-8s %(psub)6s %%         %(name)s " 
    @classmethod
    def label(cls): 
        return cls.fmt % dict(nsmp="nsmp",psmp="psmp",fsmp="fsmp",nsub="nsub",psub="psub",name="name")
    
    def __init__(self, line):
        dict.__init__(self)
        elem = line.strip().split("%")
        assert len(elem) == 4, line
        nsmp,psmp = interp_nf(elem[0],2)
        fsmp, = interp_nf(elem[1],1)
        nsub,psub = interp_nf(elem[2],2)
        name=elem[3]
        self.update(nsmp=nsmp,psmp=psmp,fsmp=fsmp,nsub=nsub,psub=psub,name=name)
    def __str__(self):
        return self.fmt % self

class Text(list):
    def __init__(self, path, opts): 
        self.opts = opts
        ptn = opts.pattern
        if not opts.pattern is None:
            opts.pattern = re.compile(opts.pattern)
        fp = open(path,"r")
        self.first = fp.readline()
        self[:] = map(Line,fp.readlines())

    def __str__(self):
        def f(ldict):
            if self.opts.pattern is None:
                match = True
            else:
                match = self.opts.pattern.search(ldict["name"])   
            pass
            return match and eval(self.opts.filter, ldict, {} )
        opts = "    ".join(["filter : %s" % self.opts.filter, "  sort : %s" % self.opts.sort, "reverse : %s" % self.opts.reverse])
        anno = [self.first, opts, "", Line.label()]
        return "\n".join(anno + map(str,sorted(filter(f,self), key=lambda _:_[self.opts.sort], reverse=self.opts.reverse))) 


def main():
    op = OptionParser(usage=__doc__)
    op.add_option("-f", "--filter", default="psub > 10 and nsmp > 0" ) 
    op.add_option("-s", "--sort",   default="psub" ) 
    op.add_option("-p", "--pattern",   default=None, help="Compile as regexp and list only lines with function names matching the pattern." ) 
    op.add_option("-r", "--reverse", action="store_true", default=False) 
    opts, args = op.parse_args()
    txt = Text(args[0],opts)
    print txt

if __name__ == '__main__':
    main()

