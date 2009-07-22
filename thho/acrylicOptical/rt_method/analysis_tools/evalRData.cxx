#define DATAPOINT 601

void evalRData(void) {

    ifstream fin1;
    fin1.open("tr.dat");

    ifstream fin2;
    fin2.open("dr.dat");

    Double_t wlContainer[DATAPOINT];
    Double_t dataContainer[DATAPOINT];
    Double_t rmsContainer[DATAPOINT];
    Double_t wlContainer1[DATAPOINT];
    Double_t dataContainer1[DATAPOINT];
    Double_t rmsContainer1[DATAPOINT];
    Double_t wlContainer2[DATAPOINT];
    Double_t dataContainer2[DATAPOINT];
    Double_t rmsContainer2[DATAPOINT];
    for(Int_t i=0;i<DATAPOINT;i++) {
        fin1 >> wlContainer1[i] >> dataContainer1[i] >> rmsContainer1[i];
        fin2 >> wlContainer2[i] >> dataContainer2[i] >> rmsContainer2[i];
        wlContainer[i] = wlContainer1[i];
        dataContainer[i] = dataContainer1[i] - dataContainer2[i];
        rmsContainer[i] = sqrt(rmsContainer1[i]*rmsContainer1[i] + rmsContainer2[i]*rmsContainer2[i]);
    }

    fin1.close();
    fin2.close();

    ofstream fout;
    fout.open("gr.dat");

    for(Int_t i=0;i<DATAPOINT;i++) {
        fout << wlContainer[i] << " " << dataContainer[i] << " " << rmsContainer[i] << endl;
    }


}
