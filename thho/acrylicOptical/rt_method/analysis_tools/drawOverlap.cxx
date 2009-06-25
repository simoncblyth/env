#define TOTALDATANO 601
#define PI 3.1415926

void drawOverlap() {

    drawParasOfFile("paras1.dat","paras2.dat");
    //drawParasOfFile("paras3.dat");


}

void drawParasOfFile(string filename, string filename2) {

    double wlArrayTmp[TOTALDATANO], nArrayTmp[TOTALDATANO], aArrayTmp[TOTALDATANO];
    double thindTArrayTmp[TOTALDATANO], thindRArrayTmp[TOTALDATANO];
    double numericalStatus[TOTALDATANO];
    ifstream fin;
    fin.open(filename.data());

    for(int i=0;i<TOTALDATANO;i++) {
        fin >> wlArrayTmp[i] >> nArrayTmp[i] >> aArrayTmp[i] >> thindTArrayTmp[i] >> thindRArrayTmp[i]
        >> numericalStatus[i];
    }


    int successCounter(0);
    for(int i =0;i<TOTALDATANO;i++){
        if(numericalStatus[i] < 0.01) successCounter++;
    }
    cout << successCounter << endl;
    const int successSize = successCounter;
    double wlArray[successSize], nArray[successSize], aArray[successSize];
    double thindTArray[successSize], thindRArray[successSize];
    int successIndex(0);
    for(int i =0;i<TOTALDATANO;i++){
        if(numericalStatus[i] < 0.01) {
            wlArray[successIndex] = wlArrayTmp[i];
            nArray[successIndex] = nArrayTmp[i];
            aArray[successIndex] = aArrayTmp[i];
            thindTArray[successIndex] =  thindTArrayTmp[i];
            thindRArray[successIndex] = thindRArrayTmp[i];
            successIndex++;
        }
    }


    /*
    for(int i=0;i<TOTALDATANO;i++) {
        //if(aArray[i] > 1.0e-5 && aArray[i] < 1.0e-4) cout << wlArray[i] << " " << aArray[i] << endl;
        if((fabs(thindTArray[i]) > 1.0e-2) || (fabs(thindRArray[i]) > 1.0e-2)) {
            cout << wlArray[i] << " " << thindTArray[i] << " " << thindRArray[i] << " " 
            << endl;
        }
    }
    */
    fin.close();

    double wlTwoArrayTmp[TOTALDATANO], nTwoArrayTmp[TOTALDATANO], aTwoArrayTmp[TOTALDATANO];
    double thindTTwoArrayTmp[TOTALDATANO], thindRTwoArrayTmp[TOTALDATANO];
    double numericalStatusTwo[TOTALDATANO];
    ifstream fin2;
    fin2.open(filename2.data());

    for(int i=0;i<TOTALDATANO;i++) {
        fin2 >> wlTwoArrayTmp[i] >> nTwoArrayTmp[i] >> aTwoArrayTmp[i] >> thindTTwoArrayTmp[i] >> thindRTwoArrayTmp[i]
        >> numericalStatusTwo[i];
    }
    /*
    for(int i=0;i<TOTALDATANO;i++) {
        //if(aTwoArray[i] > 1.0e-5 && aTwoArray[i] < 1.0e-4) cout << wlTwoArray[i] << " " << aTwoArray[i] << endl;
        if((fabs(thindTTwoArray[i]) > 1.0e-2) || (fabs(thindRTwoArray[i]) > 1.0e-2)) {
            cout << wlTwoArray[i] << " " << thindTTwoArray[i] << " " << thindRTwoArray[i] << " " 
            << endl;
        }
    }
    */
    fin2.close();

    successCounter = 0;
    for(int i =0;i<TOTALDATANO;i++){
        if(numericalStatusTwo[i] < 0.01) successCounter++;
    }
    cout << successCounter << endl;
    const int successSizeTwo = successCounter;
    double wlTwoArray[successSizeTwo], nTwoArray[successSizeTwo], aTwoArray[successSizeTwo];
    double thindTTwoArray[successSizeTwo], thindRTwoArray[successSizeTwo];
    successIndex = 0;
    for(int i =0;i<TOTALDATANO;i++){
        if(numericalStatusTwo[i] < 0.01) {
            wlTwoArray[successIndex] = wlTwoArrayTmp[i];
            nTwoArray[successIndex] = nTwoArrayTmp[i];
            aTwoArray[successIndex] = aTwoArrayTmp[i];
            thindTTwoArray[successIndex] =  thindTTwoArrayTmp[i];
            thindRTwoArray[successIndex] = thindRTwoArrayTmp[i];
            successIndex++;
        }
    }

    


    //string canvasTi = "Acrylic Optical Parameters, " + filename;

    TCanvas *c1 = new TCanvas(
        "c1","Acrylic Optical Parameters",200,10,700,900);
    c1->Divide(1,3);

    c1->cd(1);
    TH1F *hrn = c1->DrawFrame(150,1.4,850,1.8);
    hrn->SetXTitle("wavelength nm");
    hrn->SetYTitle("index of refraction");
    TGraph *grn = new TGraph(successSize, wlArray, nArray);
    grn->SetTitle("Index of Refraction V.S. Wavelength");
    grn->SetMarkerColor(kBlue);
    grn->Draw("LP");
    TGraph *grn2 = new TGraph(successSizeTwo, wlTwoArray, nTwoArray);
    grn2->SetMarkerColor(kRed);
    grn2->Draw("LP");
/*
    c1->cd(2);
    TH1F *hrnr = c1->DrawFrame(150,-0.1,850,0.1);
    hrnr->SetXTitle("wavelength nm");
    hrnr->SetYTitle("index of refraction deviation");
    double nRatioArray[TOTALDATANO];
    for(int i=0;i<TOTALDATANO;i++) nRatioArray[i] = nArray[i] - nTwoArray[i];
    TGraph *grnr = new TGraph(TOTALDATANO, wlArray, nRatioArray);
    grnr->SetMarkerColor(kGreen);
    grnr->Draw("LP");

*/
    c1->cd(2);
    TH1F *hrk = c1->DrawFrame(150,0,850,0.6);
    hrk->SetXTitle("wavelength nm");
    hrk->SetYTitle("alpha 1/mm");
    grk = new TGraph(successSize, wlArray, aArray);
    //grk->SetTitle("Alpha V.S. Wavelength");
    //grk->GetXaxis()->SetTitle("nm");
    //grk->GetYaxis()->SetTitle("alpha, 1/mm");
    grk->SetMarkerColor(kBlue);
    grk->Draw("LP");
    grk2 = new TGraph(successSizeTwo, wlTwoArray, aTwoArray);
    grk2->SetMarkerColor(kRed);
    grk2->Draw("LP");
/*
    c1->cd(4);
    TH1F *hrkr = c1->DrawFrame(150,0,850,200);
    hrkr->SetXTitle("wavelength nm");
    hrkr->SetYTitle("alpha 1/mm");
    double aRatioArray[TOTALDATANO];
    for(int i=0;i<TOTALDATANO;i++) aRatioArray[i] = aArray[i]-aTwoArray[i];
    grkr = new TGraph(TOTALDATANO, wlArray, aRatioArray);
    //grk->SetTitle("Alpha V.S. Wavelength");
    //grk->GetXaxis()->SetTitle("nm");
    //grk->GetYaxis()->SetTitle("alpha, 1/mm");
    grkr->SetMarkerColor(kBlue);
    grkr->Draw("LP");
*/


    double attArray[successSize];
    for(int i=0;i<successSize;i++) {
        attArray[i] = (1.0/aArray[i])/1000.0; // unit: m
    }
    double attTwoArray[successSizeTwo];
    for(int i=0;i<successSizeTwo;i++) {
        attTwoArray[i] = (1.0/aTwoArray[i])/1000.0; // unit: m
    }

    c1->cd(3);
    TH1F *hratt = c1->DrawFrame(150,0,850,20);
    hratt->SetXTitle("wavelength nm");
    hratt->SetYTitle("attenuation length m");
    gratt = new TGraph(successSize, wlArray, attArray);
    gratt->SetMarkerColor(kBlue);
    gratt->Draw("LP");
    gratt2 = new TGraph(successSizeTwo, wlTwoArray, attTwoArray);
    gratt2->SetMarkerColor(kRed);
    gratt2->Draw("LP");
/*
    c1->cd(6);
    TH1F *hrattr = c1->DrawFrame(150,0,850,35);
    hrattr->SetXTitle("wavelength nm");
    hrattr->SetYTitle("sample1 attnuation length / sample2 attnuation length");
    double attRatioArray[TOTALDATANO];
    for(int i=0;i<TOTALDATANO;i++) { attRatioArray[i] = attArray[i]/attTwoArray[i];}
    grattr = new TGraph(TOTALDATANO, wlArray, attRatioArray);
    grattr->Draw("LP");
*/
/*
    c1->cd(6);
    TH1F *hgrdr = c1->DrawFrame(150,-1,850,2);
    hgrdr->SetXTitle("wavelength nm");
    hgrdr->SetYTitle("Status, 0 for success, 1 for failure");
    grdr = new TGraph(TOTALDATANO, wlArray, numericalStatus);
    grdr->SetMarkerColor(kBlue);
    grdr->Draw("A*");
    grdr2 = new TGraph(TOTALDATANO, wlTwoArray, numericalStatusTwo);
    grdr2->SetMarkerColor(kRed);
    grdr2->Draw("A*");
*/


    TCanvas *c2 = new TCanvas(
        "c2","Acrylic Optical Parameters",200,10,700,900);
    c2->Divide(2,1);


    c2->cd(1);
    TH1F *hgrdt = c2->DrawFrame(150,-1,850,0.2);
    hgrdt->SetXTitle("wavelength nm");
    hgrdt->SetYTitle("calculated transmittance - measurement, %");
    for(int i=0;i<successSize;i++) { 
        thindTArray[i] = 100.0*thindTArray[i];
        if(thindTArray[i] > 0) { cout << wlArray[i] << endl;}
    }
    grdt = new TGraph(successSize, wlArray, thindTArray);
    grdt->SetMarkerColor(kBlue);
    grdt->Draw("PL");
    for(int i=0;i<successSizeTwo;i++) { 
        thindTTwoArray[i] = 100.0*thindTTwoArray[i];
        if(thindTTwoArray[i] > 0) { cout << wlArray[i] << endl;}
    }
    grdt2 = new TGraph(successSizeTwo, wlTwoArray, thindTArray);
    grdt2->SetMarkerColor(kRed);
    grdt2->Draw("PL");

/*
    c2->cd(2);
    TH1F *hgrdt = c2->DrawFrame(150,-5,850,5);
    hgrdt->SetXTitle("wavelength nm");
    hgrdt->SetYTitle("sample 1 transmittance deviation - sample 2 transmittance deviation");
    double dTRatioArray[TOTALDATANO];
    for(int i=0;i<TOTALDATANO;i++) { dTRatioArray[i] = thindTArray[i]-thindTTwoArray[i];}
    grdt = new TGraph(TOTALDATANO, wlArray, dTRatioArray);
    grdt->SetMarkerColor();
    grdt->Draw("PL");
*/
    
    c2->cd(2);
    TH1F *hgrdr = c2->DrawFrame(150,-0.1,850,0.02);
    hgrdr->SetXTitle("wavelength nm");
    hgrdr->SetYTitle("calculated reflectance - measurement, %");
    for(int i=0;i<successSize;i++) { 
        thindRArray[i] = 100.0*thindRArray[i];
        if(thindRArray[i] > 0) { cout << wlArray[i] << endl;}
    }
    grdr = new TGraph(successSize, wlArray, thindRArray);
    grdr->SetMarkerColor(kBlue);
    grdr->Draw("PL");
    for(int i=0;i<successSizeTwo;i++) {
        thindRTwoArray[i] = 100.0*thindRTwoArray[i];
        if(thindRTwoArray[i] > 0) { cout << wlArray[i] << endl;}
    }
    grdr2 = new TGraph(successSizeTwo, wlTwoArray, thindRArray);
    grdr2->SetMarkerColor(kRed);
    grdr2->Draw("PL");

}
