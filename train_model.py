import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import pickle
from sklearn.metrics import accuracy_score

# 1. Load Dataset
df = pd.read_csv('spam.csv',encoding='ISO-8859-1')  # <-- Load from CSV
#ham -> 0 -> Flase,spam ->1 ->True
df=df.rename(columns={'v1':'lable','v2':'text'})
#convert 'ham' to 0 and 'spam' to 1
df['lable']=df['lable'].map({'ham':0,'spam':1})
# 2. Check the data
print(df.head())

# 3. Text Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['text'])
y = df['lable']

# 4. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Model Training using XGBoost
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


#
# 6. Save Model and Vectorizer
with open('model.pkl', 'wb') as f:
    pickle.dump((vectorizer, model), f)

print("XGBoost model trained and saved as 'model.pkl'")