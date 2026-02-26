import pickle

with open("spam_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

print("AI Spam Detection Assistant Ready!")

while True:
    msg = input("Enter message: ")

    if msg.lower() == "exit":
        print("AI: Goodbye!")
        break


    msg_vec = vectorizer.transform([msg])
    
    prob = model.predict_proba(msg_vec)[0]
    spam_prob = prob[1]
    confidence = spam_prob * 100

    if spam_prob > 0.90:
        print(f"AI: ⚠️ SPAM ({confidence:.2f}%)\n")
    elif spam_prob > 0.70:
        print(f"AI: 🤔 Suspicious ({confidence:.2f}%)\n")
    else:
        print(f"AI: ✅ Normal ({confidence:.2f}%)\n")
        
