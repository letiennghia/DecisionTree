//
// Created by Administrator on 15-12-2024.
//
#include <bits/stdc++.h>
using namespace std;
const int max_label=3;
struct Data{
    int features[4];
    char label;
};
struct Node{
    int index;
    int threshold;
    char label;
    Node* left;
    Node* right;
    Node(){
        index = -1;
        threshold = -1;
        left = NULL;
        right = NULL;
        label = '0';
    }
};
char predict(Node *&root, Data data){
     return 'L'+rand() % max_label;
}
map<pair<char,char>, int> buildConfusionMatrix(vector<Data> data, Node *&root) {
    map<pair<char,char>,int> confussionMatrix;
    for (Data x:data) {
        char prediction=predict(root,x);
        cout<<prediction<<" "<<x.label<<endl;

        if (confussionMatrix.find(make_pair(prediction,x.label)) == confussionMatrix.end()) {
            confussionMatrix[make_pair(prediction,x.label)] = 1;
        }
        else confussionMatrix[make_pair(prediction,x.label)]++;
    }
    return confussionMatrix;
}
double CalculateF1Score(double precision, double recall) {
    if (precision + recall == 0) return 0.0;
    return 2 * (precision * recall) / (precision + recall);
}
void F1_Score(vector<Data> data, Node *&root) {
    map<pair<char,char>,int> confussionMatrix = buildConfusionMatrix(data,root);
    set<char> classes;
    for (auto x:confussionMatrix) {
        classes.insert(x.first.first);
        classes.insert(x.first.second);
    }
    map<char, double> f1Scores;
    for (auto cls:classes) {
        int TP=0, FP=0, FN=0;
        for (auto entry:confussionMatrix) {
            char predictedClass = entry.first.first;  // Predicted class
            char actualClass = entry.first.second;    // Actual class
            int count = entry.second;

            if (actualClass == cls && predictedClass == cls) {
                TP += count;  // True positives
            } else if (actualClass == cls && predictedClass != cls) {
                FN += count;  // False negatives
            } else if (actualClass != cls && predictedClass == cls) {
                FP += count;  // False positives
            }
        }
        double precision = (TP + FP == 0) ? 0.0 : static_cast<double>(TP) / (TP + FP);
        double recall = (TP + FN == 0) ? 0.0 : static_cast<double>(TP) / (TP + FN);
        double f1Score = CalculateF1Score(precision, recall);
        f1Scores[cls] = f1Score;
        cout << "Class " << cls << " -> Precision: " << precision << ", Recall: " << recall << ", F1 Score: " << f1Score << endl;
    }
    double macroF1 = 0.0;
    for (const auto& entry : f1Scores) {
        macroF1 += entry.second;
    }
    macroF1 /= f1Scores.size();

    cout << "Macro F1 Score: " << macroF1 << endl;
}
vector<Data> read_data(const string &filename) {
    vector<Data> data;
    ifstream file(filename);
    if (!file.is_open()) {
        cout<<"File Not Found"<<endl;
        return data;
    }
    string line;
    while (getline(file,line)) {

        vector<int> features;
        for (char c:line) {
            Data d;
            if (c>='A' && c<='Z') {
                d.label = c;
            }
            if (c>='0' && c<='9') {
                features.push_back(c-'0');
            }
            for (int i=0;i<features.size();i++) {
                d.features[i] = features[i];
            }
            data.push_back(d);
        }
    }
    return data;
}
int main() {
  /*Node *root = NULL;
  vector<Data> dummyData;
  for(int i=0;i<10;i++){
      Data d;
      d.label = rand() % max_label+'L';
      dummyData.push_back(d);
  }

  F1_Score(dummyData,root);*/
    vector<Data> data;
    data = read_data("train.txt");
    for (Data x:data) {
        cout<<x.label<<" "<<x.features[0]<<" "<<x.features[1]<<" "<<x.features[2]<<endl;
    }
}