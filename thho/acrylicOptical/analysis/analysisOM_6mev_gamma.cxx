gStyle->SetOptFit(1111);

TH1F *gh_peGen_all = new TH1F("peGen_all", "pe of a gamma in AD",500, 0, 1400);
TH1F *gh_peGen_GdLS = new TH1F("peGen_GdLS", "pe of a gamma (in GdLS)",500, 0, 1400);
//TH1F *gh_peGenCap_GdLS = new TH1F("peGenCap_GdLS","pe of a gamma in target",500, 0, 1400);
//TH1F *gh_peGen_iav = new TH1F("peGen_iav", "pe of a gamma (in IAV)", 500, 0, 1400);
//TH1F *gh_peCap_iav = new TH1F("peCap_iav", "pe of a gamma (in IAV)", 500, 0, 1400);
TH1F *gh_peGenCap_GdLS_iav = new TH1F("peGenCap_GdLS_iav", "pe of a gamma in IAV", 500, 0, 1400);


void analysisOM_6mev_gamma(void) {

    printRes("Resolution of a 6 Mev gamma in AD", genPlot("peGen_all", gh_peGen_all));
    printRes("Resolution of a 6 MeVgamma in target", genPlot("peGen_GdLS", gh_peGen_GdLS));
    genPlot("peGenCap_GdLS_iav",gh_peGenCap_GdLS_iav);
    printEff("Gammas stopping at iav", gh_peGen_GdLS, gh_peGenCap_GdLS_iav);
    gSystem->Exit(0);

}

void printEff(TString title, TH1F *gen, TH1F *cap) {

    cout << "********************************************************************************" << endl;
    cout << title << endl;
    cout << "Efficiency(%)\tCapture in Acrylics\t Gen in GdLS" << endl;
    cout << (cap->GetEntries()/gen->GetEntries())*100.0 << "\t" << cap->GetEntries() << "\t" << gen->GetEntries() << endl;
    cout << "********************************************************************************" << endl;

}

void printRes(TString title, TF1 *genFit) {

    cout << "********************************************************************************" << endl;
    cout << title << endl;
    cout << "Resolution(%)\tMean\tSigma" << endl;
    cout << (genFit->GetParameter(2)/genFit->GetParameter(1))*100.0 << "\t" << genFit->GetParameter(1) << "\t" << genFit->GetParameter(2) << endl;
    cout << "********************************************************************************" << endl;

}

TF1* genPlot(TString term, TH1F *gh) {

    FILE* pipe = gSystem->OpenPipe("ls GammaBasicPlots_?.root" , "r" );
    scanAnalysis(pipe, term, gh);

    TF1 *gf = new TF1("gf","gaus",600,1000);
    gf->SetLineColor(2);

    TCanvas *c = new TCanvas;
    gh->SetAxisRange(0,1200);
    gh->SetAxisRange(0,50,"Y");
    //gh->Fit("gaus");
    gh->Fit("gf","R");
    gh->Draw();

    TImage *img = TImage::Create();
    img->FromPad(c);
    TString plotName = term;
    plotName = plotName + ".png";
    img->WriteImage(plotName.Data());
    
    gSystem->ClosePipe( pipe );
    return gh->GetFunction("gf");

}


void scanAnalysis(FILE *pipe, TString term, TH1F *gh) {
    TString path ;
    while( path.Gets(pipe) )
    {
        cout << path << endl;
        f = TFile::Open( path );
        f->cd("basics");
        d = f->GetDirectory("basics");
        TH1F *h;    
        d->GetObject(term.Data(),h);

        //cout << "h add : " << h << endl;
        //cout << "d add : " << d << endl;

        gh->Add(h);
        f->Close();
    }
    

}

