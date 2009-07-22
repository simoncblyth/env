#define DATASIZE 3606
#define DATAPOINT 601

void evalData(void) {

    ifstream fin;
    fin.open("out.dat");

    Double_t wlContainer[DATASIZE];
    Double_t dataContainer[DATASIZE];
    for(Int_t i=0;i<DATASIZE;i++) fin >> wlContainer[i] >> dataContainer[i];

    ofstream fout;
    fout.open("meanData.dat", ios_base::app);

    for(Int_t i=0;i<DATAPOINT;i++) {
        TH1D *hd = new TH1D("hd","hd",102,-2,100);
        for(Int_t j=0;j<6;j++) {
            Int_t k = i + DATAPOINT*j;
            hd->Fill(dataContainer[k]);
        }
        Double_t mean = hd->GetMean();
        Double_t rms = hd->GetRMS();
        fout << wlContainer[i] << " " << mean << " " << rms << endl;
        delete hd;
    }


}
