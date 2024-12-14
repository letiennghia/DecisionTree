#include<bits/stdc++.h>
using namespace std;
const int numFeat=4;
const int maxFeatVal=5;
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
struct Data{
  int feature[numFeat];
  char label;
      Data() {
            for(int i=0;i<numFeat;i++) {
                  feature[i] = -1;
            }
            label = '0';
      }
};
vector<Data> dummyData;

double getEntropy(vector<Data> data){
      //Given the data, you need to get the label, count it then return the entropy
      map<char,int> count_of;
      for (Data x:data) {
            count_of[x.label]++;
      }
      double entropy = 0;
      for (auto x_i:count_of) {
            double probability=x_i.second / double (count_of.size());
            entropy += probability * (log(probability)/log(2));
      }
      return -entropy;
}

double getInfoGain(vector<Data> data,vector<Data> data_left, vector<Data> data_right) {
      double ratioLeft = double(data_left.size())/double(data.size());
      double ratioRight = 1-ratioLeft;
      return getEntropy(data)-(getEntropy(data_left)*ratioLeft+getEntropy(data_right)*ratioRight);
}
pair<vector<Data>,vector<Data>> splitData(vector<Data> data,int index, int threshold) {
      vector<Data> data_left,data_right;
      for (Data x:data) {
            if (x.feature[index]<=threshold) {
                  data_left.push_back(x);
            }
            else data_right.push_back(x);
      }
      return make_pair(data_left,data_right);
}

char getLabel(vector<Data> &data) {
      map<char,int> count_label;
      for (Data x:data) {
            count_label[x.label]++;
      }
      int max_label = 0;
      char max_label_char = '0';
      for (auto x_i:count_label) {
            if (max_label<x_i.second) {
                  max_label_char = x_i.first;
            }
      }
      return max_label_char;
}
void buildTree(Node *& parent, vector<Data> data, int depth, int max_depth, int min_size) {
      //check criterion
      if (depth > max_depth||data.size()<min_size||getEntropy(data)==0.0) {
            return;
      }
      vector<Data> data_left,data_right;
      double maxGain=-100;
      //find best split
      for (int i=0;i<numFeat;i++) {
            for (int j=1;j<=maxFeatVal;j++) {
                  data_left.clear();data_right.clear();
                  pair<vector<Data>,vector<Data>> split_data = splitData(data,i,j);
                  data_left = split_data.first;
                  data_right = split_data.second;
                  double gainVal=getInfoGain(data_left,data_right,data_left);
                  if (maxGain<gainVal) {
                        maxGain = gainVal;
                        parent -> index = i;
                        parent -> threshold = j;
                  }
            }
      }
      // add bestsplit to build new node
      pair<vector<Data>,vector<Data>> split_data = splitData(data,parent->index,parent->threshold);
      data_left = split_data.first;
      data_right = split_data.second;
      buildTree(parent->left,data_left,depth+1,max_depth,min_size);
      buildTree(parent->right,data_right,depth+1,max_depth,min_size);


}
int main(){
      dummyData.clear();
      for (int i=0;i<10;i++) {
            Data d;
            for (int i=0;i<numFeat;i++) {
                  d.feature[i] = rand()%3;
            }
            d.label = '0'+rand()%3;
            dummyData.push_back(d);
      }
      vector<Data> data_left,data_right;
      pair<vector<Data>,vector<Data>> data = splitData(dummyData,0,1);
      data_left = data.first;
      data_right = data.second;
      cout<<getInfoGain(dummyData,data_right,data_left)<<endl;
      for (Data d:data_left) {
            cout<<d.label<<" "<<d.feature[0]<<" "<<d.feature[1]<<" "<<d.feature[2]<<endl;
      }
      cout<<endl;
      for (Data d:data_right) {
            cout<<d.label<<" "<<d.feature[0]<<" "<<d.feature[1]<<" "<<d.feature[2]<<endl;
      }
}