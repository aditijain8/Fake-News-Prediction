# Fake-News-Prediction
This project is a machine learning-based system to detect fake news using Natural Language Processing (NLP) techniques. It uses the Logistic Regression algorithm for classification after processing and vectorizing text data from news articles.

🔧Technologies Used
</br>
Python
</br>
Pandas & NumPy
</br>
NLTK (Natural Language Toolkit)
</br>
Scikit-learn (sklearn)
</br>
TfidfVectorizer
</br>
Logistic Regression
</br>

📁 Dataset
</br>
https://www.kaggle.com/c/fake-news/data?select=train.csv
</br>
The dataset used is a labeled CSV file (train.csv) with the following columns:
</br>
id: Unique identifier for each article</br>
title: Title of the news article</br>
author: Author of the article</br>
text: Full text of the article</br>
label: Binary label (1 = Fake, 0 = Real)</br>

🧠 Workflow
</br>
1. Import Dependencies
All required libraries like pandas, nltk, and sklearn are imported. Stopwords from NLTK are also downloaded.
2. Load Dataset
python
Copy
Edit
news_dataset = pd.read_csv('/content/train.csv')
Dataset shape: (20800, 5)
3. Data Cleaning
Missing values are filled with empty strings.
author and title columns are merged to create a new content column.
4. Text Preprocessing
Using regular expressions, lowercase conversion, stopword removal, and Porter Stemming, the text is cleaned and normalized.
python
Copy
Edit
def stemming(content):
5. Feature Extraction
Text data is converted into numerical features using TF-IDF Vectorization.
python
Copy
Edit
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)
6. Train-Test Split
python
Copy
Edit
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
7. Model Training
A Logistic Regression model is trained on the vectorized features.
python
Copy
Edit
model = LogisticRegression()
model.fit(X_train, Y_train)
8. Evaluation
The model's accuracy is tested on both training and test sets using accuracy_score.
</br>
</br>
📊 Results
</br>
You can evaluate the model using:
python
Copy
Edit
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
</br>
</br>
🚀 How to Run
</br>
Clone the repository or download the notebook.
</br>
Install required libraries:
bash
Copy
Edit
pip install numpy pandas scikit-learn nltk
Run the Jupyter notebook or Python script.
Make sure train.csv is available in the working directory.
</br>
</br>
📌 Future Improvements
</br>
Use advanced models like Random Forest or XGBoost
Use deep learning (LSTM, BERT)
Build a web interface for real-time prediction
