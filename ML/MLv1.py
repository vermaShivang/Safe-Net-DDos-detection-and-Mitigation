from datetime import datetime
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

class MachineLearning():
    def __init__(self):
        print("Loading dataset ...")
        self.counter = 0
        self.flow_dataset = pd.read_csv('FlowStatsfile.csv')
        self.flow_dataset.iloc[:, 2] = self.flow_dataset.iloc[:, 2].str.replace('.', '')
        self.flow_dataset.iloc[:, 3] = self.flow_dataset.iloc[:, 3].str.replace('.', '')
        self.flow_dataset.iloc[:, 5] = self.flow_dataset.iloc[:, 5].str.replace('.', '')

        self.X_flow = self.flow_dataset.iloc[:, :-1].values.astype('float64')
        self.y_flow = self.flow_dataset.iloc[:, -1].values
        self.X_flow_train, self.X_flow_test, self.y_flow_train, self.y_flow_test = train_test_split(self.X_flow, self.y_flow, test_size=0.25, random_state=0)

        os.makedirs("output", exist_ok=True)

    def LR(self):
        self.classifier = LogisticRegression(solver='liblinear', random_state=0)
        self.Confusion_matrix("Logistic_Regression")

    def KNN(self):
        self.classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
        self.Confusion_matrix("KNN")

    def SVM(self):
        self.classifier = SVC(kernel='rbf', random_state=0)
        self.Confusion_matrix("SVM")

    def NB(self):
        self.classifier = GaussianNB()
        self.Confusion_matrix("Naive_Bayes")

    def DT(self):
        self.classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
        self.Confusion_matrix("Decision_Tree")

    def RF(self):
        self.classifier = RandomForestClassifier(n_estimators=10, criterion="entropy", random_state=0)
        self.Confusion_matrix("Random_Forest")

    def Confusion_matrix(self, model_name):
        self.counter += 1
        model = self.classifier.fit(self.X_flow_train, self.y_flow_train)
        y_pred = model.predict(self.X_flow_test)

        cm = confusion_matrix(self.y_flow_test, y_pred)
        acc = accuracy_score(self.y_flow_test, y_pred)
        fail = 1.0 - acc

        print(f"{model_name} Accuracy: {acc:.2f} | Failure: {fail:.2f}")
        print(cm)

        # Heatmap Plot
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Pred 0', 'Pred 1'], yticklabels=['True 0', 'True 1'])
        plt.title(f"{model_name} - Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.tight_layout()
        heatmap_path = f"output/{model_name}_Confusion_Matrix.png"
        plt.savefig(heatmap_path)
        plt.close()

        # Bar Plot
        x_labels = ['TP', 'FP', 'FN', 'TN']
        y_values = [cm[0][0], cm[0][1], cm[1][0], cm[1][1]]
        x_indexes = np.arange(len(x_labels))
        plt.figure(figsize=(6, 4))
        plt.bar(x_indexes, y_values, width=0.4, color='skyblue')
        plt.xticks(x_indexes, x_labels)
        plt.title(f"{model_name} - Results of the Algorithm")
        plt.xlabel("Prediction Outcome")
        plt.ylabel("Number of Flows")
        plt.tight_layout()
        result_path = f"output/{model_name}_Results.png"
        plt.savefig(result_path)
        plt.close()

def main():
    start_script = datetime.now()
    ml = MachineLearning()

    # for model_func in [ ml.NB, ml.DT, ml.RF]:
    for model_func in [ml.SVM]:

    # for model_func in [ml.LR, ml.KNN, ml.SVM, ml.NB, ml.DT, ml.RF]:
        start = datetime.now()
        model_func()
        end = datetime.now()
        print(f"Time Taken: {(end - start)}")
    
    print("Total Script Time:", datetime.now() - start_script)

if __name__ == "__main__":
    main()
