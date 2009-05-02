void csvToData(void) {

    FILE* pipe = gSystem->OpenPipe("ls *.csv" , "r" );
    TString finname;
    while(finname.Gets(pipe)) {
    cout << finname << endl;
    TString foutname = "compartment_" + finname;
    convertFormat(finname,foutname);
    }

}

void convertFormat(TString inputFile, TString outputFile) {


    ifstream fin;
    fin.open(inputFile.Data());

    ofstream fout;
    fout.open(outputFile.Data());

    TString col_1, col_2;
    TString comma=",";

    while(1) {
        if(!fin.good()) break;
        fin >> col_1 >> col_2;
        if(col_1 != "nm," && col_2 != "%T") {
            Int_t colEnd = col_1.Length();
            col_1.Remove(colEnd-1);
            fout << col_1 << " " << col_2 << endl;
        }
    }

    fin.close();
    fout.close();


}
