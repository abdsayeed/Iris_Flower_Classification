🌸 Iris Flower Classification
Machine Learning Classification Project



A machine learning project that classifies iris flowers into three species using their physical measurements.
Multiple classification algorithms are trained and compared for performance.
📑 Table of Contents
Project Overview
Dataset
Tech Stack
Workflow
Models Used
Results
Visualizations
Installation & Usage
Project Structure
Future Improvements
License
📌 Project Overview
This project demonstrates a complete machine learning pipeline:
✔ Data loading and cleaning
✔ Exploratory data analysis (EDA)
✔ Data visualization
✔ Feature encoding
✔ Model training and evaluation
✔ Multi-model comparison
The model predicts iris species based on:
Sepal Length
Sepal Width
Petal Length
Petal Width
🎯 Target Classes
Iris-setosa
Iris-versicolor
Iris-virginica
📂 Dataset
File: Iris.csv
Total samples: 150
Features: 4 numerical
Target: 1 categorical
Balanced classes (50 samples each)
⚙️ Tech Stack
Category	Tools
Language	Python
Data Handling	Pandas, NumPy
Visualisation	Matplotlib, Seaborn
Machine Learning	Scikit-learn
Environment	Jupyter Notebook
🔍 Workflow
1️⃣ Data Loading
Import dataset using pandas
Remove unnecessary columns
2️⃣ Data Exploration
Summary statistics
Data types
Class distribution
Missing values
3️⃣ Data Visualisation
Histograms
Scatter plots
Correlation heatmap
4️⃣ Data Preprocessing
Label encoding
Train-test split (70/30)
5️⃣ Model Training
Logistic Regression
K-Nearest Neighbours
Decision Tree
6️⃣ Model Evaluation
Accuracy score comparison
🤖 Models Used
Model	Purpose
Logistic Regression	Linear classification baseline
KNN	Distance-based classification
Decision Tree	Rule-based classification
📊 Results
All models achieved perfect accuracy on the test dataset.
Model	Accuracy
Logistic Regression	100%
KNN	100%
Decision Tree	100%
📈 Visualisations
✔ Feature distribution histograms
✔ Species scatter plots
✔ Correlation heatmap
These help understand class separability and feature relationships.
▶️ Installation & Usage
Clone repository
git clone https://github.com/your-username/iris-classification.git
cd iris-classification
Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn
Run project
Open Jupyter Notebook and run all cells.
📁 Project Structure
iris-classification/
│
├── Iris.csv
├── iris_model.ipynb
├── README.md
🚀 Future Improvements
Cross-validation
Hyperparameter tuning
Confusion matrix
Precision / Recall / F1 score
Model deployment (Streamlit / Flask)
📜 License
This project is for educational and learning purposes.
