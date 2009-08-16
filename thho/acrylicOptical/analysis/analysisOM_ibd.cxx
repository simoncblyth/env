//gStyle->SetOptFit(1111);
#define NCUT 810.323
#define NMAX 3000.0

TH1F *gh_nHits = new TH1F("nHits", "Neutron Capture Hits in GdLS", 100, 0, 1400);
TH1F *gh_nHits_onGd = new TH1F("nHits_onGd", "on-Gd nCap Hits", 100, 0, 1400);


void analysisOM_ibd(void) {

    printRes("Neutron detection resolution in target", genPlotNeutron("nHits", gh_nHits));
    genPlotEff("nHits_onGd",gh_nHits_onGd);
    printEff("Neutron capture on Gd", gh_nHits_onGd);
    gSystem->Exit(0);

}

void printEff(TString title, TH1F *cap) {

    cap->SetAxisRange(NCUT, NMAX);

    cout << "********************************************************************************" << endl;
    cout << title << endl;
    cout << "Efficiency(%)\tNeutron Capture on Gd with Cut\t Neutron Capture on Gd" << endl;
    cout << (cap->GetEffectiveEntries()/cap->GetEntries())*100.0 << "\t" << cap->GetEffectiveEntries() << "\t" << cap->GetEntries() << endl;
    cout << "********************************************************************************" << endl;

}

void printRes(TString title, TF1 *genFit) {

    cout << "********************************************************************************" << endl;
    cout << title << endl;
    cout << "Resolution(%)\tMean\tSigma" << endl;
    cout << (genFit->GetParameter(2)/genFit->GetParameter(1))*100.0 << "\t" << genFit->GetParameter(1) << "\t" << genFit->GetParameter(2) << endl;
    cout << (genFit->GetParameter(5)/genFit->GetParameter(4))*100.0 << "\t" << genFit->GetParameter(4) << "\t" << genFit->GetParameter(5) << endl;
    cout << "********************************************************************************" << endl;

}

TF1* genPlotNeutron(TString term, TH1F *gh) {

    FILE* pipe = gSystem->OpenPipe("ls IbdBasicPlots_?.root" , "r" );
    scanAnalysis(pipe, term, gh);

    TCanvas *c = new TCanvas;
    Double_t par[6];
    TF1 *g1    = new TF1("g1","gaus",0,500);
    TF1 *g2    = new TF1("g2","gaus",500,1400);
    TF1 *total = new TF1("total","gaus(0)+gaus(3)",0,1400);
    total->SetLineColor(2);
    gh->Fit(g1,"R");
    gh->Fit(g2,"R+");
    g1->GetParameters(&par[0]);
    g2->GetParameters(&par[3]);
    total->SetParameters(par);
    gh->Fit(total,"R+");

    TImage *img = TImage::Create();
    img->FromPad(c);
    TString plotName = term;
    plotName = plotName + ".png";
    img->WriteImage(plotName.Data());
    
    gSystem->ClosePipe( pipe );
    return gh->GetFunction("total");

}

void genPlotEff(TString term, TH1F *gh) {

    FILE* pipe = gSystem->OpenPipe("ls IbdBasicPlots_?.root" , "r" );
    scanAnalysis(pipe, term, gh);

    TCanvas *c = new TCanvas;
    TF1 *g2    = new TF1("g2","gaus",500,1400);
    g2->SetLineColor(2);
    gh->SetAxisRange(0,1400);
    gh->SetAxisRange(0,100,"Y");
    gh->Fit("g2","R");
    gh->Draw();

    TImage *img = TImage::Create();
    img->FromPad(c);
    TString plotName = term;
    plotName = plotName + ".png";
    img->WriteImage(plotName.Data());

    gSystem->ClosePipe( pipe );
}

void scanAnalysis(FILE *pipe, TString term, TH1F *gh) {
    TString path ;
    while( path.Gets(pipe) )
    {
        cout << path << endl;
        f = TFile::Open( path );
        d = f->GetDirectory("stats");
        dd = d->GetDirectory("basics");

        TH1F *h;    
        dd->GetObject(term.Data(),h);

        //cout << "h add : " << h << endl;
        //cout << "d add : " << d << endl;

        gh->Add(h);
        f->Close();
    }
    

}

