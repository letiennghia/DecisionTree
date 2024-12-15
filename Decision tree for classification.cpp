#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>
#include <fstream>
#include <sstream>
#include <unordered_map>

using namespace std;

// Define a structure for a Node in the Decision Tree
struct Node {
    bool isLeaf;
    int featureIndex;
    double threshold;
    double gini;
    double entropy;
    vector<Node*> children;
    int classLabel;
};

// Calculate Gini impurity
// double giniImpurity(const vector<int>& labels) {
//     int count = labels.size();
//     if (count == 0) return 0.0;
//     vector<int> labelCount(3, 0); //3 labels: R, B, L
//     for (int label : labels) {
//         labelCount[label]++;
//     }
//     double gini = 1.0;
//     for (int i = 0; i < labelCount.size(); ++i) {
//         double proportion = static_cast<double>(labelCount[i]) / count;
//         gini -= proportion * proportion;
//     }
//     return gini;
// }

// Calculate Entropy
double entropy(const vector<int>& labels) {
    int count = labels.size();
    if (count == 0) return 0.0;
    vector<int> labelCount(3, 0);
    for (int label : labels) {
        labelCount[label]++;
    }
    double entropy = 0.0;
    for (int i = 0; i < labelCount.size(); ++i) {
        if (labelCount[i] == 0) continue;
        double proportion = static_cast<double>(labelCount[i]) / count;
        entropy -= proportion * log2(proportion);
    }
    return entropy;
}

// Split data based on feature and threshold
void splitData(const vector<vector<double>>& data, const vector<int>& labels, int featureIndex, double threshold,
               vector<vector<double>>& leftData, vector<int>& leftLabels,
               vector<vector<double>>& rightData, vector<int>& rightLabels) {
    for (size_t i = 0; i < data.size(); ++i) {
        if (data[i][featureIndex] <= threshold) {
            leftData.push_back(data[i]);
            leftLabels.push_back(labels[i]);
        } else {
            rightData.push_back(data[i]);
            rightLabels.push_back(labels[i]);
        }
    }
}

// Find the best split based on Gini impurity or entropy
tuple<int, double, double> findBestSplit(const vector<vector<double>>& data, const vector<int>& labels) {
    int bestFeatureIndex = -1;
    double bestThreshold = 0.0;
    double bestEntropy = numeric_limits<double>::max();

    for (size_t featureIndex = 0; featureIndex < data[0].size(); ++featureIndex) {
        vector<double> thresholds;
        for (const auto& sample : data) {
            thresholds.push_back(sample[featureIndex]);
        }
        sort(thresholds.begin(), thresholds.end());

        for (size_t i = 1; i < thresholds.size(); ++i) {
            double threshold = (thresholds[i - 1] + thresholds[i]) / 2.0;
            vector<vector<double>> leftData, rightData;
            vector<int> leftLabels, rightLabels;
            splitData(data, labels, featureIndex, threshold, leftData, leftLabels, rightData, rightLabels);

            //double leftGini = giniImpurity(leftLabels);
            //double rightGini = giniImpurity(rightLabels);
            double leftEntropy = entropy(leftLabels);
            double rightEntropy = entropy(rightLabels);
            double weightedEntropy = (leftLabels.size() * leftEntropy + rightLabels.size() * rightEntropy) / labels.size();

            if (weightedEntropy < bestEntropy) {
                bestEntropy = weightedEntropy;
                bestFeatureIndex = featureIndex;
                bestThreshold = threshold;
            }
        }
    }

    return make_tuple(bestFeatureIndex, bestThreshold, bestEntropy);
}

// Decision Tree Node Creation
Node* createNode() {
    Node* node = new Node();
    node->isLeaf = false;
    node->featureIndex = -1;
    node->threshold = 0.0;
    node->gini = 0.0;
    node->entropy = 0.0;
    node->classLabel = -1;
    return node;
}

// Build Tree Recursively
Node* buildTree(const vector<vector<double>>& data, const vector<int>& labels, int depth, int maxDepth) {
    if (depth >= maxDepth || data.empty() || entropy(labels) == 0.0) {
        // Create a leaf node
        Node* leaf = createNode();
        leaf->isLeaf = true;
        leaf->classLabel = (labels.size() > 0) ? labels[0] : -1;  // Assign majority label
        return leaf;
    }

    // Find the best split
    auto [bestFeatureIndex, bestThreshold, bestEntropy] = findBestSplit(data, labels);
    if (bestFeatureIndex == -1) {
        // Create a leaf node if no valid split is found
        Node* leaf = createNode();
        leaf->isLeaf = true;
        leaf->classLabel = (labels.size() > 0) ? labels[0] : -1;
        return leaf;
    }

    // Create internal node
    Node* node = createNode();
    node->featureIndex = bestFeatureIndex;
    node->threshold = bestThreshold;
    node->entropy = bestEntropy;

    // Split data into left and right
    vector<vector<double>> leftData, rightData;
    vector<int> leftLabels, rightLabels;
    splitData(data, labels, bestFeatureIndex, bestThreshold, leftData, leftLabels, rightData, rightLabels);


    // Check for pruning condition
    double pruningThreshold = 0.03; 
    if (leftLabels.empty() || rightLabels.empty() || entropy(labels) - bestEntropy < pruningThreshold) {
        Node* leaf = createNode();
        leaf->isLeaf = true;
        // Assign the majority label to this leaf
        unordered_map<int, int> labelCount;
        for (int label : labels) {
            labelCount[label]++;
        }
        leaf->classLabel = max_element(labelCount.begin(), labelCount.end(),
                                    [](const pair<int, int>& a, const pair<int, int>& b) {
                                        return a.second < b.second;
                                    })->first;

        return leaf;
    }

    // Recursively build children
    node->children.push_back(buildTree(leftData, leftLabels, depth + 1, maxDepth));
    node->children.push_back(buildTree(rightData, rightLabels, depth + 1, maxDepth));

    return node;
}

// Predict for a given input
int predict(Node* node, const vector<double>& input) {
    if (node->isLeaf) {
        return node->classLabel;
    }
    if (input[node->featureIndex] <= node->threshold) {
        return predict(node->children[0], input);
    } else {
        return predict(node->children[1], input);
    }
}

// Convert label from int to string
string labelToString(int label) {
    switch (label) {
        case 0: return "R";
        case 1: return "B";
        case 2: return "L";
        default: return "Unknown";
    }
}

// Convert label from string to int
int labelToInt(const string& labelStr) {
    if (labelStr == "R") return 0;
    if (labelStr == "B") return 1;
    if (labelStr == "L") return 2;
    return -1;
}

// Calculate F1 Score
double f1Score(const vector<int>& actualLabels, const vector<int>& predictedLabels) {
    int tp = 0, fp = 0, fn = 0;
    for (int i = 0; i < actualLabels.size(); ++i) {
        if (actualLabels[i] == predictedLabels[i] && actualLabels[i] != -1) tp++;
        else if (actualLabels[i] != predictedLabels[i] && predictedLabels[i] != -1) fp++;
        else if (actualLabels[i] != predictedLabels[i] && predictedLabels[i] == -1) fn++;
    }
    double precision = (tp + fp == 0) ? 0 : tp / static_cast<double>(tp + fp);
    double recall = (tp + fn == 0) ? 0 : tp / static_cast<double>(tp + fn);

    if (precision == 0 && recall == 0) {
        return 0.0; // avoid divied by 0
    }

    return 2 * (precision * recall) / (precision + recall);
}

// Cross-validation to find the best hyperparameters
pair<int, double> crossValidation(const vector<vector<double>>& data, const vector<int>& labels, int maxDepth) {
    int bestMaxDepth = 1;
    double bestF1Score = 0.0;

    for (int depth = 1; depth <= maxDepth; ++depth) {
        // Split data into training and validation sets (simple 80-20 split for demonstration)
        size_t splitIndex = data.size() * 0.8;
        vector<vector<double>> trainData(data.begin(), data.begin() + splitIndex);
        vector<int> trainLabels(labels.begin(), labels.begin() + splitIndex);
        vector<vector<double>> valData(data.begin() + splitIndex, data.end());
        vector<int> valLabels(labels.begin() + splitIndex, labels.end());

        // Build tree
        Node* root = buildTree(trainData, trainLabels, 0, depth);

        // Predict on validation set
        vector<int> predictedLabels;
        for (const auto& input : valData) {
            predictedLabels.push_back(predict(root, input));
        }

        // Calculate F1 Score
        double f1 = f1Score(valLabels, predictedLabels);
        if (f1 > bestF1Score) {
            bestF1Score = f1;
            bestMaxDepth = depth;
        }
    }

    return make_pair(bestMaxDepth, bestF1Score);
}

// Read data from a file
void readData(const string& filename, vector<vector<double>>& data, vector<int>& labels, bool hasLabel) {
    ifstream file(filename);
    if (!file.is_open()) {
        cout << "Error: Could not open file " << filename << endl;
        return;
    }

    string line;
    while (getline(file, line)) {
        stringstream ss(line);
        if (hasLabel) {
            string labelStr;
            getline(ss, labelStr, ',');
            int label = labelToInt(labelStr);
            if (label == -1) continue;
            labels.push_back(label);
        }

        vector<double> features;
        string featureStr;
        while (getline(ss, featureStr, ',')) {
            features.push_back(stod(featureStr));
        }
        data.push_back(features);
    }

    file.close();
    if (labels.empty() && hasLabel) {
        cout << "Warning: No labels were read from file " << filename << endl;
    }
}

int main() {
    // Input data and labels from train_data file
    vector<vector<double>> trainData;
    vector<int> trainLabels;
    readData("train.txt", trainData, trainLabels, true);

    // Parameters
    int maxDepth = 20;
    int minSamplesSplit = 3;

    // Cross-validation to find best maxDepth
    auto [bestMaxDepth, bestF1] = crossValidation(trainData, trainLabels, maxDepth);
    cout << "Best Max Depth: " << bestMaxDepth << ", Best F1 Score: " << bestF1 << endl;

    // Build decision tree with best maxDepth
    Node* root = buildTree(trainData, trainLabels, 0, bestMaxDepth);

    // Predict labels for test_data.txt
    vector<vector<double>> testData;
    vector<int> testPredictedLabels;
    readData("test.txt", testData, testPredictedLabels, false);
    ofstream outputFile("Predict_labels.txt");
    if (!outputFile.is_open()) {
        cout << "Error: Could not open file to write predicted labels." << endl;
        return 1;
    }
    for (const auto& input : testData) {
        int predictedLabel = predict(root, input);
        testPredictedLabels.push_back(predictedLabel);
        outputFile << labelToString(predictedLabel) << endl;
        cout << "Predicted Label: " << labelToString(predictedLabel) << endl;
    }
    outputFile.close();

    // Read actual labels from test_data have labels
    vector<vector<double>> testDataLabel;
    vector<int> testActualLabels;
    readData("result.txt", testDataLabel, testActualLabels, true);

    // Calculate F1 Score for test data
    double testF1Score = f1Score(testActualLabels, testPredictedLabels);
    cout << "F1 Score on Test Data: " << testF1Score << endl;

    return 0;
}
