
import os
from ROOT import gROOT

## how to control the dir in which the .so gets created ?
gROOT.ProcessLine(".L %s/TEveDigitSet_Additions.cxx+" % os.path.dirname(__file__) )

from ROOT import TEveDigitSet_GetDigitValue, TEveDigitSet_SetDigitValue, TEveDigitSet_PrintDigit
from ROOT import TEveDigitSet_SetDigitColorI, TEveDigitSet_SetDigitColorIT, TEveDigitSet_SetDigitColorRGBA
from ROOT import TEveDigitSet
TEveDigitSet.PrintDigit     = TEveDigitSet_PrintDigit
TEveDigitSet.GetDigitValue  = TEveDigitSet_GetDigitValue
TEveDigitSet.SetDigitValue  = TEveDigitSet_SetDigitValue

TEveDigitSet.SetDigitColorI     = TEveDigitSet_SetDigitColorI
TEveDigitSet.SetDigitColorIT    = TEveDigitSet_SetDigitColorIT
TEveDigitSet.SetDigitColorRGBA  = TEveDigitSet_SetDigitColorRGBA


