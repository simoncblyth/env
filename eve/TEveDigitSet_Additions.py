
from ROOT import gROOT
gROOT.ProcessLine(".L TEveDigitSet_Additions.cxx+")
from ROOT import TEveDigitSet_GetDigitValue, TEveDigitSet_SetDigitValue, TEveDigitSet_PrintDigit, TEveDigitSet_SetDigitColor
from ROOT import TEveDigitSet
TEveDigitSet.PrintDigit     = TEveDigitSet_PrintDigit
TEveDigitSet.GetDigitValue  = TEveDigitSet_GetDigitValue
TEveDigitSet.SetDigitValue  = TEveDigitSet_SetDigitValue
TEveDigitSet.SetDigitColor  = TEveDigitSet_SetDigitColor


