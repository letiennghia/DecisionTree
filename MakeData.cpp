//
// Created by letie on 12/15/2024.
//
#include<bits/stdc++.h>
using namespace std;
int main(){
    string filename="Data/train.txt";
    ifstream file(filename);
    if (!file.is_open()) {
        cout<<"File Not Found"<<endl;
        return 0;
    }
    freopen("Data/trainRL.txt", "w", stdout);
    string line;
    while (getline(file,line)) {
         if(line[0]=='R'||line[0]=='L'){
           cout<<line<<endl;
         }
    }
}