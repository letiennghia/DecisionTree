#include<bits/stdc++.h>

#include <random>
using namespace std;
const int numFeat=4;
const int maxFeatVal=5;
const double minGainVal=0.02;
//build Node
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
//build Data
struct Data{
      char label;
      int feature[numFeat];
};

// Calculating entropy function
double getEntropy(vector<Data> data){
      //Given the data, you need to get the label, count it then return the entropy
      map<char,int> count_of;
      for (Data x:data) {
            count_of[x.label]++;
      }
      double entropy = 0;
      for (auto x_i:count_of) {
            double probability=(double) x_i.second / data.size();
            entropy -= probability * (log(probability)/log(2));
      }
      return entropy;
}
// Information gain
double getInfoGain(vector<Data> data,vector<Data> data_left, vector<Data> data_right) {
      if (data_left.empty() || data_right.empty()) {return 0.0;}
      double ratioLeft = (double)data_left.size()/data.size();
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
                  max_label = x_i.second;
                  max_label_char = x_i.first;
            }
      }
      return max_label_char;
}
void buildTree(Node*& parent, vector<Data> data, int depth, int max_depth, int min_size) {
      parent = new Node();
      // Check termination conditions
      if (depth >= max_depth || data.size() <= min_size || getEntropy(data) == 0.0) {
            parent->label = getLabel(data);
            parent->left = NULL;
            parent->right = NULL;
            return;
      }

      vector<Data> data_left, data_right;
      double maxGain = -10;
      // Find best split
      for (int i = 0; i < numFeat; i++) {

            for (int j = 0; j <= maxFeatVal; j++) {
                  pair<vector<Data>, vector<Data>> split_data = splitData(data, i, j);
                  data_left.clear();
                  data_right.clear();
                  data_left = split_data.first;
                  data_right = split_data.second;
                  if (data_left.empty() || data_right.empty()) {continue;}
                  double gainVal = getInfoGain(data, data_left, data_right);
                  //cout<<gainVal<<endl;
                  if (maxGain < gainVal) {
                        maxGain = gainVal;
                        parent->index = i;
                        parent->threshold = j;
                  }
            }
      }
     // cout << "Building Tree at depth " << depth << ", data size: " << data.size() << endl;
     // cout << "Best split: index=" << parent->index << " threshold=" << parent->threshold << endl;
      if ()
      // Split data
      pair<vector<Data>, vector<Data>> split_data = splitData(data, parent->index, parent->threshold);
      data_left = split_data.first;
      data_right = split_data.second;

      buildTree(parent->left, data_left, depth + 1, max_depth, min_size);
      buildTree(parent->right, data_right, depth + 1, max_depth, min_size);
      if(parent -> left -> label != '0' && parent -> left -> label == parent -> right -> label){
            parent->label = parent->left->label;
            parent->left = NULL;
            parent->right = NULL;
      }
}
char predict(Node *&parent, Data data) {
      if (parent -> left == NULL) {
            return parent->label;
      }
      if (data.feature[parent->index]<=parent->threshold) {
            return predict(parent->left,data);
      }
      else return predict(parent->right,data);
}
map<pair<char,char>, int> buildConfusionMatrix(vector<Data> data, Node *&root) {
      map<pair<char,char>,int> confussionMatrix;
      for (Data x:data) {
            char prediction=predict(root,x);
            if (confussionMatrix.find(make_pair(prediction,x.label)) == confussionMatrix.end()) {
                  confussionMatrix[make_pair(prediction,x.label)] = 1;
            }
            else confussionMatrix[make_pair(prediction,x.label)]++;
      }
      /*char val[3]={'L','R','B'};
      for (char c1:val) {
            for (char c2:val) cout<<confussionMatrix[make_pair(c1,c2)]<<" ";
            cout<<endl;
      }
      cout<<endl;*/
      return confussionMatrix;

}
double CalculateF1Score(double precision, double recall) {
      if (precision + recall == 0) return 0.0;
      return 2 * (precision * recall) / (precision + recall);
}
double F1_Score(vector<Data> data, Node *&root) {
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
            //cout << "Class " << cls << " -> Precision: " << precision << ", Recall: " << recall << ", F1 Score: " << f1Score << endl;

      }
      double macroF1 = 0.0;
      for (const auto& entry : f1Scores) {
            macroF1 += entry.second;
      }
      macroF1 /= f1Scores.size();
      //cout << "Macro F1 Score: " << macroF1 << endl;
      return macroF1;
}
void printTree(Node *&parent, int depth) {
      if (parent==NULL) {return;}
      if (parent -> left == NULL||depth>=100) {
            cout<<parent->label<<" "<<depth<<endl;
      }
      cout<<"L2"<<endl;
      cout<<parent->index<<" "<<parent->threshold<<" "<<parent->label<<endl;
      cout<<"left"<<endl;
      printTree(parent->left,depth+1);
      printTree(parent->right,depth+1);

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
            Data d;
            for (char c:line) {

                  if (c==',') continue;
                  if (c>='A' && c<='Z') {
                        //cout<<c<<" ";
                        d.label = c;
                  }
                  else if (c>='0' && c<='9') {
                        //cout<<c<<" ";
                        features.push_back(c-'0');
                  }

            }
            for (int i=0;i<features.size();i++) {
                  d.feature[i] = features[i];
            }
            data.push_back(d);
            //cout<<d.label<<" "<<d.feature[0]<<" "<<d.feature[1]<<" "<<d.feature[2]<<" "<<d.feature[3]<<endl;
      }
      return data;
}
void shuffleData(vector<Data> &data) {
      shuffle(data.begin(), data.end(), std::mt19937(std::random_device()()));
}
double train_data(vector<Data> &train_data,vector<Data> &test_data, const int&maxDepth,const int&minSize) {

      Node *root = NULL;
      buildTree(root,train_data,0,maxDepth,minSize);
      return F1_Score(test_data,root);
}
tuple<double, int, int> crossValidation(vector<Data> data) {
      shuffleData(data);  // Shuffle the data before splitting
      int n = data.size();
      double best_solution = 0.0;
      int best_max_depth = 1, best_min_size = 0;
      double bestF1Score = 0.0;
      size_t splitIndex=n*0.8;
      vector<Data> trainData(data.begin(), data.begin()+splitIndex);
      vector<Data> testData(data.begin()+splitIndex,data.end());
      //cout<<train_data(trainData,testData,20,20)<<endl;
      //cout<<trainData.size()<<endl;
      //cout<<testData.size()<<endl;
      for (int i=1;i<=30;i++) {
            for (int j=20;j>=1;j--)
                  if (bestF1Score<train_data(trainData,testData,i,j)) {
                        bestF1Score = train_data(trainData,testData,i,j);
                        //cout<<bestF1Score<<" "<<train_data(trainData,testData,i,j)<<endl;
                        best_max_depth = i;
                        best_min_size = j;
                  }

      }
      cout << "Best F1: " << bestF1Score << endl;
      cout << "Best max_depth: " << best_max_depth << endl;
      cout << "Best min_size: " << best_min_size << endl;
      return make_tuple(best_solution, best_max_depth, best_min_size);
}


int main(){
      vector<Data> data=read_data("train.txt");
      //cout<<train_data(data,data,20,1)<<endl;
      for (int i=0;i<100;i++) {
            tuple<double,int,int>result=crossValidation(data);

      }
      //cout<<get<0>(result)<<endl;
      //cout<<get<1>(result)<<endl;
      //cout<<get<2>(result)<<endl;
      return 0;

      for (Data d:data) {
            cout<<d.label<<" "<<d.feature[0]<<" "<<d.feature[1]<<" "<<d.feature[2]<<" "<<d.feature[3]<<endl;
      }

      cout<<train_data(data,data,5,5);
      return 0;
}