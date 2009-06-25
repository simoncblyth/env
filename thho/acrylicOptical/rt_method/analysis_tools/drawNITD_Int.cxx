#define TOTALDATANO 601
void drawNITD_Int(void) {



    double RMSArray_1[TOTALDATANO], RMSArray_2[TOTALDATANO], RMSArray_3[TOTALDATANO], RMSArray_4[TOTALDATANO];
    double MEANArray_1[TOTALDATANO],MEANArray_2[TOTALDATANO], MEANArray_3[TOTALDATANO], MEANArray_4[TOTALDATANO];
    double wl[TOTALDATANO];

    drawFromFile("compartment_10-1-1.csv","compartment_10-1-2.csv","compartment_10-1-3.csv","compartment_10-1-4.csv","compartment_10-1-5.csv", RMSArray_1, MEANArray_1, wl);
    drawFromFile("compartment_Sample2.Sample.Cycle1.Raw.csv","compartment_Sample2.Sample.Cycle2.Raw.csv","compartment_Sample2.Sample.Cycle3.Raw.csv","compartment_Sample2.Sample.Cycle4.Raw.csv","compartment_Sample2.Sample.Cycle5.Raw.csv",RMSArray_2,MEANArray_2,wl);
    drawFromFile("compartment_Sample3.Sample.Cycle1.Raw.csv","compartment_Sample3.Sample.Cycle2.Raw.csv","compartment_Sample3.Sample.Cycle3.Raw.csv","compartment_Sample3.Sample.Cycle4.Raw.csv","compartment_Sample3.Sample.Cycle5.Raw.csv",RMSArray_3,MEANArray_3,wl);
    drawFromFile("int_Sample15.Sample.Cycle1.Raw.csv","int_Sample15.Sample.Cycle2.Raw.csv","int_Sample15.Sample.Cycle3.Raw.csv","int_Sample15.Sample.Cycle4.Raw.csv","int_Sample15.Sample.Cycle5.Raw.csv",RMSArray_4,MEANArray_4,wl);



    double diffMEAN_1[TOTALDATANO], diffMEAN_2[TOTALDATANO], diffMEAN_3[TOTALDATANO];
    double diffRMS_1[TOTALDATANO], diffRMS_2[TOTALDATANO], diffRMS_3[TOTALDATANO];
    double wlError[TOTALDATANO];

    for(int i=0;i<TOTALDATANO;i++) {

        diffMEAN_1[i] = MEANArray_1[i] - MEANArray_2[i];
        diffMEAN_2[i] = MEANArray_3[i] - MEANArray_2[i];
        diffMEAN_3[i] = MEANArray_4[i] - MEANArray_2[i];
        diffRMS_1[i] = RMSArray_1[i]*RMSArray_1[i] + RMSArray_2[i]*RMSArray_2[i];
        diffRMS_2[i] = RMSArray_3[i]*RMSArray_3[i] + RMSArray_2[i]*RMSArray_2[i];
        diffRMS_3[i] = RMSArray_4[i]*RMSArray_4[i] + RMSArray_2[i]*RMSArray_2[i];


        wlError[i] = 0.06;

    }



    TCanvas *c_1 = new TCanvas("c_1", "reproducibitity of diff locations in compartment",200,10,700,900);


    TH1F *hr = c_1->DrawFrame(150,-3,850,3);
    hr->SetXTitle("wavelength nm");
    hr->SetYTitle("delta T, %");

    TGraphErrors *gre_1 = new TGraphErrors(TOTALDATANO, wl, diffMEAN_1, wlError, diffRMS_1);
    gre_1->SetMarkerColor(kGreen);
    gre_1->SetLineColor(kGreen);
    gre_1->Draw("LP");

    TGraphErrors *gre_2 = new TGraphErrors(TOTALDATANO, wl, diffMEAN_2, wlError, diffRMS_2);
    gre_2->SetMarkerColor(kBlue);
    gre_2->SetLineColor(kBlue);
    gre_2->Draw("LP");

    TGraphErrors *gre_3 = new TGraphErrors(TOTALDATANO, wl, diffMEAN_3, wlError, diffRMS_3);
    gre_3->SetMarkerColor(kRed);
    gre_3->SetLineColor(kRed);
    gre_3->Draw("LP");



/*
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
*/

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

