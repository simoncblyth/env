// an quick drawing using the formula 2 in DocDB2570v3
//

void DrawAbsLen(Double_t cutting, Double_t a1, Double_t a2, Double_t a3) {

    TCanvas *c1 = new TCanvas(
        "c1","Optical Model AbsLength and T",200,10,700,900);
    title = new TPaveText(.2,0.96,.8,.995);
    title->AddText("Optical Model AbsLength and T");
    title->Draw();

    pad1 = new TPad("pad1","AbsLengthPad",0.03,0.02,0.98,0.48,21);
    pad2 = new TPad("pad2","TransmittancePad",0.03,0.50,0.98,0.95,21);
    pad1->Draw();
    pad2->Draw();

    TF1 *f1 = new TF1("ABsLength",GetAbs,0,800,4);
    f1->SetParameters(cutting,a1,a2,a3);
    f1->Draw();




}

Double_t GetAbs(Double_t *x, Double_t *par) {

    return ((par[1]-par[2])/(1+exp((x[0]-par[0])/par[3])))+par[2];

}
