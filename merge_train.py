import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ----------------------------
# LOAD DATASET 1
# ----------------------------
df1 = pd.read_csv("dataset1.csv")
df1 = df1.dropna()
df1["text"] = df1["text"].astype(str)

df1 = df1[["label", "text"]]

# ----------------------------
# LOAD DATASET 2
# ----------------------------
df2 = pd.read_csv("dataset2.csv")
df2 = df2.dropna()
df2["Message"] = df2["Message"].astype(str)

# convert ham/spam → 0/1
df2["label"] = df2["Class"].map({
    "ham": 0,
    "spam": 1
})

df2 = df2.rename(columns={"Message": "text"})
df2 = df2[["label", "text"]]

# ----------------------------
# MERGE BOTH
# ----------------------------
df = pd.concat([df1, df2], ignore_index=True)

print("Total dataset:", df.shape)
print(df["label"].value_counts())

# ----------------------------
# VECTORIZE
# ----------------------------
vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1,2),
    min_df=2,
    max_df=0.95
)

X = vectorizer.fit_transform(df["text"])
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ----------------------------
# MODEL
# ----------------------------
model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ----------------------------
# SAVE
# ----------------------------
pickle.dump(model, open("spam_model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("\nModel trained and saved!")