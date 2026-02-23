🌸 Iris Flower Classification – Machine Learning Project
This project builds a machine learning model to classify iris flowers into three species based on their physical measurements. It uses the classic Iris dataset and compares multiple classification algorithms to evaluate performance.
📌 Project Overview
The goal of this project is to:
Explore and understand the Iris dataset
Perform data preprocessing and visualization
Train multiple machine learning models
Compare model performance
Predict iris species based on flower measurements
The classification is based on:
Sepal Length
Sepal Width
Petal Length
Petal Width
Target classes:
Iris-setosa
Iris-versicolor
Iris-virginica
📂 Dataset
The dataset used is Iris.csv, which contains 150 samples of iris flowers with 4 numerical features and 1 categorical target variable.
Each class has 50 samples.
⚙️ Technologies Used
Python
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn
Jupyter Notebook
🔍 Project Workflow
1️⃣ Data Loading
Load dataset using pandas
Remove unnecessary columns (Id)
2️⃣ Data Exploration
Statistical summary
Data types and structure
Class distribution
Null value check
3️⃣ Data Visualization
Histograms for feature distribution
Scatter plots for feature relationships
Correlation heatmap
4️⃣ Data Preprocessing
Label encoding for species classification
Train-test split (70% training, 30% testing)
5️⃣ Model Training
The following models were trained:
Logistic Regression
K-Nearest Neighbours (KNN)
Decision Tree
6️⃣ Model Evaluation
Accuracy score used for performance comparison.
📊 Results
All models achieved 100% accuracy on the test set:
Model	Accuracy
Logistic Regression	100%
KNN	100%
Decision Tree	100%
▶️ How to Run the Project
Clone the repository
git clone https://github.com/your-username/iris-classification.git
Navigate to project folder
cd iris-classification
Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn
Run the Jupyter notebook or Python script
📈 Visualizations Included
Feature distribution histograms
Scatter plots between feature pairs
Correlation heatmap
These help understand feature relationships and class separability.
🎯 Key Learnings
Basic data preprocessing techniques
Feature visualization and analysis
Multi-class classification
Model comparison in machine learning
Using scikit-learn for training and evaluation
🚀 Possible Improvements
Cross-validation for more reliable evaluation
Hyperparameter tuning
Model performance comparison with metrics (precision, recall, F1)
Confusion matrix visualization
Deployment as a web app
📜 License
This project is open-source and free to use for educational purposes.
