

from ROOT import gROOT, kTRUE, kFALSE
gROOT.ProcessLine(".L TEveDigitSet_Additions.cxx+")
from ROOT import TEveDigitSet_GetValue, TEveDigitSet_SetValue, TEveDigitSet_PrintValue

from ROOT import TRandom, TEveQuadSet, TNamed
r = TRandom(0)
q = TEveQuadSet("RectangleXY")
q.Print()
q.Reset(TEveQuadSet.kQT_RectangleXY, kFALSE, 32)

num = 10 
for i in range(num):
    q.AddQuad(r.Uniform(-10, 9), r.Uniform(-10, 9), 0, r.Uniform(0.2, 1), r.Uniform(0.2, 1))
    q.QuadValue(100+i)
    tn = TNamed("QuadIdx %d" % i , "TNamed assigned to a quad as an indentifier.")
    q.QuadId(tn)
   
q.RefitPlex()

for i in range(num): TEveDigitSet_PrintValue( q , i )
print " ".join( [ "%d" % TEveDigitSet_GetValue( q , i ) for i in range(num) ])
for i in range(num):TEveDigitSet_SetValue( q , i , 200+i )
print " ".join( [ "%d" % TEveDigitSet_GetValue( q , i ) for i in range(num) ])
for i in range(num):assert TEveDigitSet_GetValue( q , i ) == 200 + i
for i in range(num):TEveDigitSet_PrintValue( q , i )





