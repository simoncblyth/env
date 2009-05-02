void csvToData(void) {

    FILE* pipe = gSystem->OpenPipe("ls *.cvs" , "r" );
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
            //col_1.replace(col_1.find(comma),comma.length(),"");
            fout << col_1 << " " << col_2 << endl;
        }
    }

    fin.close();
    fout.close();


}
