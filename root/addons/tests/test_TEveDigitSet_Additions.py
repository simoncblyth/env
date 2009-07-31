import ROOT
ROOT.PyConfig.GUIThreadScheduleOnce += [ ROOT.TEveManager.Create ]


from ROOT import kTRUE, kFALSE
import TEveDigitSet_Additions

def make_quad( colIsVal, num=10 ):
    from ROOT import TRandom, TEveQuadSet, TNamed, TColor
    r = TRandom(0)
    q = TEveQuadSet("RectangleXY")
    q.Reset(TEveQuadSet.kQT_RectangleXY, colIsVal, 32)
    for i in range(num):
        q.AddQuad(r.Uniform(-10, 9), r.Uniform(-10, 9), 0, r.Uniform(0.2, 1), r.Uniform(0.2, 1))
        if colIsVal:
            ci = 1001 + i
            tc = TColor( ci , r.Uniform(0.1,1), r.Uniform(0.1,1), r.Uniform(0.1, 1), "mycol%s" % i , r.Uniform(0.1, 1))
            q.QuadColor(ci)
        else:
            q.QuadValue(100+i)
        tn = TNamed("QuadIdx %d" % i , "TNamed assigned to a quad as an indentifier.")
        q.QuadId(tn)
    q.RefitPlex()
    return q
    

def test_quad( q ):
    colIsVal = q.GetValueIsColor()  
    num = q.GetPlex().Size()

    for i in range(num):
        q.PrintDigit( i )

    if colIsVal:
        pass
    else:
        print " ".join( [ "%d" % q.GetDigitValue( i ) for i in range(num) ])

        for i in range(num):
            q.SetDigitValue( i , 200+i )

        print " ".join( [ "%d" % q.GetDigitValue( q , i ) for i in range(num) ])

        for i in range(num):
            assert q.GetDigitValue( i ) == 200 + i

        for i in range(num):
            q.PrintDigit(i)


if __name__=='__main__':
    ROOT.PyGUIThread.finishSchedule()

    q = make_quad( kTRUE )
    test_quad( q )

    from ROOT import gEve
    gEve.AddElement(q)
    gEve.Redraw3D(kTRUE)

