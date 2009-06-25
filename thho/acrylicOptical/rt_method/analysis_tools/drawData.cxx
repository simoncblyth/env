#define TOTALDATANO 601
void drawData(void) {



    double RMSArray_1[TOTALDATANO], RMSArray_2[TOTALDATANO];
    double MEANArray_1[TOTALDATANO],MEANArray_2[TOTALDATANO];
    double wl[TOTALDATANO];

    drawFromFile("1-1-1-1.csv","1-1-1-2.csv","1-1-1-3.csv", RMSArray_1, MEANArray_1, wl);
    //drawFromFile("1-1-2-1.csv","1-1-2-2.csv","1-1-2-3.csv",RMSArray_2,MEANArray_2,wl);



    TCanvas *c_1 = new TCanvas("c_1", "reproducibitity",200,10,700,900);
    c_1->Divide(2,2);

    c_1->cd(1);
    TH1F *h_1 = c_1->DrawFrame(150,0.0,850,100.0);
    h_1->SetXTitle("wavelength nm");
    h_1->SetYTitle("Transmittance Mean, %");
    TGraph *gr_1 = new TGraph(TOTALDATANO, wl, MEANArray_1);
    gr_1->SetTitle("Transmittance v.s. wavelength");
    gr_1->SetMarkerColor(kBlue);
    gr_1->Draw("PL");
    TGraph *gr_3 = new TGraph(TOTALDATANO, wl, MEANArray_3);
    gr_3->SetMarkerColor(kRed);
    gr_3->Draw("PL");


    c_1->cd(2);
    TH1F *h_2 = c_1->DrawFrame(150,0.0,850,10.0);
    h_2->SetXTitle("wavelength nm");
    h_2->SetYTitle("Reflectance Mean, %");
    TGraph *gr_2 = new TGraph(TOTALDATANO, wl, MEANArray_2);
    gr_2->SetTitle("Reflectance v.s. wavelength");
    gr_2->SetMarkerColor(kBlue);
    gr_2->Draw("PL");
    TGraph *gr_4 = new TGraph(TOTALDATANO, wl, MEANArray_4);
    gr_4->SetMarkerColor(kRed);
    gr_4->Draw("PL");

    c_1->cd(3);
    TH1F *h_3 = c_1->DrawFrame(150,0.0,850,0.2);
    h_3->SetXTitle("wavelength nm");
    h_3->SetYTitle("Transmittance RMS, %");
    //h_3->SetTitleOffset(5.0);
    TGraph *gr_5 = new TGraph(TOTALDATANO, wl, RMSArray_1);
    gr_5->SetTitle("Reflectance v.s. wavelength");
    gr_5->SetMarkerColor(kBlue);
    gr_5->Draw("PL");
    TGraph *gr_6 = new TGraph(TOTALDATANO, wl, RMSArray_3);
    gr_6->SetMarkerColor(kRed);
    gr_6->Draw("PL");


    c_1->cd(4);
    TH1F *h_4 = c_1->DrawFrame(150,0.0,850,0.2);
    h_4->SetXTitle("wavelength nm");
    h_4->SetYTitle("Reflectance RMS, %");
    TGraph *gr_7 = new TGraph(TOTALDATANO, wl, RMSArray_2);
    gr_7->SetTitle("Reflectance v.s. wavelength");
    gr_7->SetMarkerColor(kBlue);
    gr_7->Draw("PL");
    TGraph *gr_8 = new TGraph(TOTALDATANO, wl, RMSArray_4);
    gr_8->SetMarkerColor(kRed);
    gr_8->Draw("PL");


}

void drawFromFile(string file1, string file2, string file3, string file4, string file5, double RMSArray[], double MEANArray[], double wl[]) {

    ifstream fin1, fin2, fin3, fin4, fin5;
    double wl1[TOTALDATANO], wl2[TOTALDATANO], wl3[TOTALDATANO], wl4[TOTALDATANO], wl5[TOTALDATANO];
    double data1[TOTALDATANO], data2[TOTALDATANO], data3[TOTALDATANO], data4[TOTALDATANO], data5[TOTALDATANO];



    fin1.open(file1.data());
    fin2.open(file2.data());
    fin3.open(file3.data());
    fin4.open(file4.data());
    fin5.open(file5.data());

    for(int i=0;i<TOTALDATANO;i++) {

        TH1D *h = new TH1D("h","htitle",1000,0,100);
        fin1 >> wl1[i] >> data1[i];
        fin2 >> wl2[i] >> data2[i];
        fin3 >> wl3[i] >> data3[i];
        fin4 >> wl4[i] >> data4[i];
        fin5 >> wl5[i] >> data5[i];
        h->Fill(data1[i]);
        h->Fill(data2[i]);
        h->Fill(data3[i]);
        h->Fill(data4[i]);
        h->Fill(data5[i]);
        RMSArray[i] = h->GetRMS();
        MEANArray[i] = h->GetMean();
        delete h;
    }
    fin1.close();
    fin2.close();
    fin3.close();
    fin4.close();
    fin5.close();


    for(int i=0;i<TOTALDATANO;i++) wl[i] = wl1[i];



/*
    c1->cd(1);
    TGraph *gr = new TGraph(TOTALDATANO, wl1, RMSArray);
    gr->Draw("APL");

    c1->cd(2);
    TGraph *gr2 = new TGraph(TOTALDATANO, wl1, MEANArray);
    gr2->Draw("APL");
*/

    //h->Draw();

}

