#define WLCUT 280
#define ATTCUT 10000
#define TOTALDATANO 601
#define PI 3.1415926


void drawParas() {

    drawParasOfFile("paras.dat");
    //drawParasOfFile("paras2.dat");
    //drawParasOfFile("paras3.dat");


}

void drawParasOfFile(string filename) {

//    double wlArray[TOTALDATANO], nArray[TOTALDATANO], aArray[TOTALDATANO];
//    double thindTArray[TOTALDATANO], thindRArray[TOTALDATANO];
    double wlArrayTmp[TOTALDATANO], nArrayTmp[TOTALDATANO], nArrayErrTmp[TOTALDATANO];
    double aArrayTmp[TOTALDATANO], aArrayErrTmp[TOTALDATANO];
    double attArrayTmp[TOTALDATANO], attArrayUpErrTmp[TOTALDATANO], attArrayLowErrTmp[TOTALDATANO];
    double thindTArrayTmp[TOTALDATANO], thindRArrayTmp[TOTALDATANO];
    int numericalStatus[TOTALDATANO];
    ifstream fin;
    fin.open(filename.data());

    for(int i=0;i<TOTALDATANO;i++) {
        fin >> wlArrayTmp[i] >> nArrayTmp[i] >> nArrayErrTmp[i] >> 
                aArrayTmp[i] >> aArrayErrTmp[i] >>
                attArrayTmp[i] >> attArrayUpErrTmp[i] >> attArrayLowErrTmp[i] >>
                thindTArrayTmp[i] >> thindRArrayTmp[i] >> numericalStatus[i];
    }

    double numericalStatusGra[TOTALDATANO];
    for(int i=0;i<TOTALDATANO;i++) { numericalStatusGra[i] = numericalStatus[i];}


    int sc_1(0);
    int sc_2(0);
    int sc_3(0);
    for(int i=0;i<TOTALDATANO;i++) {
        if(numericalStatus[i] != 0 && wlArrayTmp[i] > 450.0) sc_1++;
        if(numericalStatus[i] != 0 && wlArrayTmp[i] < 451.0 && wlArrayTmp[i] > 300.0) sc_2++;
        if(numericalStatus[i] != 0 && wlArrayTmp[i] < 301.0) {
            cout << wlArrayTmp[i] << endl;
            sc_3++;
        }
    }
    cout << " > 450 failed : " << sc_1 << endl;
    cout << " > 300 and < 451 failed : " << sc_2 << endl;
    cout << " < 300 failed : " << sc_3 << endl;


    // do not draw fail status
    int successStatus(0);
    for(int i=0;i<TOTALDATANO;i++) {
        if(numericalStatus[i] == 0 && wlArrayTmp[i] > WLCUT && (attArrayUpErrTmp[i] > 1.0) && ((attArrayTmp[i] + attArrayUpErrTmp[i]) < 10000.0)) {
        //if(numericalStatus[i] == 0 && wlArrayTmp[i] > WLCUT) {
            successStatus++;
        }
    }
    const int successSize = successStatus;
    double wlArray[successSize], nArray[successSize], aArray[successSize], attArray[successSize];
    double wlArrayErr[successSize], nArrayErr[successSize], aArrayErr[successSize];
    double attArrayUpErr[successSize], attArrayLowErr[successSize];
    double thindTArray[successSize], thindRArray[successSize];
    int ns[TOTALDATANO];
    for(int i=0;i<TOTALDATANO;i++) ns[i] = 1;

    cout << "su size " << successSize << endl;

    int suCounter(0);
    for(int i=0;i<TOTALDATANO;i++) {
        if(numericalStatus[i] == 0 && wlArrayTmp[i] > WLCUT && (attArrayUpErrTmp[i] > 1.0) && ((attArrayTmp[i] + attArrayUpErrTmp[i]) < 10000.0)) {
        //if(numericalStatus[i] == 0 && wlArrayTmp[i] > WLCUT) {
            wlArray[suCounter] = wlArrayTmp[i];
            wlArrayErr[suCounter] = 0.5;
            nArray[suCounter] = nArrayTmp[i];
            aArray[suCounter] = aArrayTmp[i];
            attArray[suCounter] = attArrayTmp[i];
            nArrayErr[suCounter] = nArrayErrTmp[i];
            aArrayErr[suCounter] = aArrayErrTmp[i];
            attArrayUpErr[suCounter] = attArrayUpErrTmp[i];
            //cout << attArrayUpErr[i] << endl;
            attArrayLowErr[suCounter] = attArrayLowErrTmp[i];
            thindTArray[suCounter] = thindTArrayTmp[i];
            thindRArray[suCounter] = thindRArrayTmp[i];
            ns[i] = 0;
            suCounter++;
        }
    }


    string canvasTi = "Acrylic Optical Parameters, " + filename;

    TCanvas *c1 = new TCanvas(
        filename.data(),canvasTi.data(),200,10,700,900);
    c1->Divide(2,3);
    
    c1->cd(1);
    //grn = new TGraphErrors(successSize, wlArray, nArray, wlArrayErr, nArrayErr);
    grn = new TGraph(successSize, wlArray, nArray);
    grn->SetTitle("Index of Refraction V.S. Wavelength");
    grn->GetXaxis()->SetTitle("nm");
    grn->GetYaxis()->SetTitle("n");
    grn->SetMarkerColor(kRed);
    grn->SetLineColor(kBlue);
    grn->Draw("A*");

    c1->cd(2);
    //grk = new TGraphErrors(successSize, wlArray, aArray, wlArrayErr, aArrayErr);
    grk = new TGraph(successSize, wlArray, aArray);
    grk->SetTitle("Alpha V.S. Wavelength");
    grk->GetXaxis()->SetTitle("nm");
    grk->GetYaxis()->SetTitle("alpha, 1/mm");
    grk->SetMarkerColor(kRed);
    grk->SetLineColor(kBlue);
    //grk->SetMarkerStyle(21);
    grk->Draw("A*");

    //double attArray[successSize];
    //for(int i=0;i<successSize;i++) {
    //    attArray[i] = (1.0/aArray[i])/1000.0; // unit: m
    //}

    c1->cd(3);
    //gratt = new TGraphAsymmErrors(successSize, wlArray, attArray, wlArrayErr, wlArrayErr, attArrayLowErr, attArrayUpErr);
    gratt = new TGraph(successSize, wlArray, attArray);
    gratt->SetTitle("Attenuation V.S. Wavelength");
    gratt->GetXaxis()->SetTitle("nm");
    gratt->GetYaxis()->SetTitle("mm");
    gratt->SetLineColor(kBlue);
    gratt->SetMarkerColor(kRed);
    //gratt->SetMarkerStyle(21);
    gratt->Draw("A*");

    c1->cd(4);
    grdr = new TGraph(TOTALDATANO, wlArrayTmp, numericalStatusGra);
    grdr->SetTitle("Numerical Status V.S. Wavelength");
    grdr->GetXaxis()->SetTitle("nm");
    grdr->GetYaxis()->SetTitle("Status, 0:Success, 1:failded");
    //grdr->SetMarkerStyle(21);
    grdr->SetMarkerColor(kRed);
    grdr->Draw("A*");

    c1->cd(5);
    grdt = new TGraph(successSize, wlArray, thindTArray);
    grdt->SetTitle("Transmittance Deviation V.S. Wavelength, Thin sample");
    grdt->GetXaxis()->SetTitle("nm");
    grdt->GetYaxis()->SetTitle("dT");
    grdt->SetLineColor(kBlue);
    grdt->SetMarkerColor(kRed);
    //grdt->SetMarkerStyle(21);
    grdt->Draw("A*");

    c1->cd(6);
    grdr = new TGraph(successSize, wlArray, thindRArray);
    grdr->SetTitle("Reflectance Deviation V.S. Wavelength, Thin sample");
    grdr->GetXaxis()->SetTitle("nm");
    grdr->GetYaxis()->SetTitle("dR");
    grdr->SetLineColor(kBlue);
    grdr->SetMarkerColor(kRed);
    //grdr->SetMarkerStyle(21);
    grdr->Draw("A*");

    /*
    string cc = "Solution status " + filename;
    string cc2 = filename+"2";
    TCanvas *c2 = new TCanvas(
        cc2.data(),cc.data(),200,10,700,900);

    grdr2 = new TGraph(TOTALDATANO, wlArrayTmp, numericalStatusGra);
    grdr2->SetTitle("Numerical Status V.S. Wavelength");
    grdr2->GetXaxis()->SetTitle("nm");
    grdr2->GetYaxis()->SetTitle("Status, 0:Success, 1:failded");
    //grdr->SetMarkerStyle(21);
    if(filename == "paras.dat") grdr2->SetMarkerColor(kBlue);
    if(filename == "paras2.dat") grdr2->SetMarkerColor(kRed);
    grdr2->Draw("A*");
    */

}
