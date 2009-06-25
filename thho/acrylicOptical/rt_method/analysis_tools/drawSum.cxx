#define TOTALDATANO 601
void drawSum(void) {

    drawFromFile("1-1-1-1.csv_1-1-1-2.csv_1-1-1-3.csv","1-1-2-1.csv_1-1-2-2.csv_1-1-2-3.csv");
    drawFromFile("2-1-1-1.csv_2-1-1-2.csv_2-1-1-3.csv","2-1-2-1.csv_2-1-2-2.csv_2-1-2-3.csv");

}


void drawFromFile(string tFile, string rFile) {


    ifstream tFin, rFin;
    ofstream fout;
    string outName = tFile + "_" + rFile;
    fout.open(outName.data());
    tFin.open(tFile.data());
    rFin.open(rFile.data());
    double tWl[TOTALDATANO], rWl[TOTALDATANO], tData[TOTALDATANO], rData[TOTALDATANO];
    double tRms[TOTALDATANO], rRms[TOTALDATANO];
    double sumData[TOTALDATANO], sumRms[TOTALDATANO];

    for(int i=0;i<TOTALDATANO;i++){

        tFin >> tWl[i] >> tData[i] >> tRms[i];
        rFin >> rWl[i] >> rData[i] >> rRms[i];

        sumData[i] = tData[i] + rData[i];
        sumRms[i] = sqrt(tRms[i]*tRms[i] + rRms[i]*rRms[i]);
        cout << tWl[i] << " " << sumData[i] << " " <<  sumRms[i] << endl;

        fout << tWl[i] << " " << sumData[i] << " " <<  sumRms[i] << endl;

    }

    fout.close();

    TCanvas *c_1 = new TCanvas("c_1","Sum of RT",200,10,700,900);

    gr = new TGraph(TOTALDATANO,tWl,sumData);
    gr->Draw("A*");


}
