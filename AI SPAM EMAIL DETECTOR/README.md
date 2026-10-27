# 🛡️ AI Spam Email & SMS Detector

A complete, beginner-friendly, and interactive college demonstration project for detecting spam and phishing emails using **Natural Language Processing (NLP)** and **Machine Learning**.

Built with **Python**, **Streamlit**, and **scikit-learn**, utilizing **TF-IDF text vectorization** and a **Multinomial Naive Bayes (MultinomialNB)** classifier.

---

## 👨‍💻 Developer Credit
**Developed by Prathmesh Khedkar**

---

## 🌟 Key Features

1. **Interactive Streamlit Web Dashboard**:
   - Modern, responsive user interface with glassmorphism styling and custom CSS badges.
   - Large input area for testing any email subject line, email body, or SMS message.
   - Quick-load sample buttons for instant testing (Phishing, Lottery Scam, Delivery Scam, College Assignment, Team Meeting, Flight Ticket).
   
2. **Transparent ML Predictions & Feature Explanations**:
   - Predicts **SPAM** or **NOT SPAM (LEGITIMATE)**.
   - Displays real posterior confidence probabilities (e.g. 96.5% Spam vs 3.5% Ham).
   - **Model Interpretability Breakdown**: Shows key terms in the input message matching the training vocabulary, their TF-IDF weights, and whether each term influenced the prediction towards Spam or Legitimate.

3. **Authentic Scikit-Learn Pipeline**:
   - Built on a real scikit-learn `Pipeline([('tfidf', TfidfVectorizer(...)), ('nb', MultinomialNB(...))])`.
   - No hardcoded keyword rules or simulated logic.
   - Trained on a structured demonstration dataset (`dataset.csv`) containing 100 balanced, realistic examples.

4. **Rigorous Train / Test Evaluation**:
   - 80/20 train/test split with stratified sampling.
   - Computes and displays true held-out test metrics:
     - **Accuracy**
     - **Precision**
     - **Recall**
     - **F1-Score**
     - **Interactive Confusion Matrix** (True Positives, False Positives, True Negatives, False Negatives)

5. **College Viva & Theory Guide**:
   - In-depth, simplified mathematical and conceptual explanations for:
     - TF-IDF ($TF \times IDF$) formulation and intuition.
     - Multinomial Naive Bayes and the conditional independence assumption.
     - Laplace Smoothing ($\alpha$) and the zero-frequency problem.
     - Step-by-step training and inference workflow.
     - Common viva questions with model answers.
   - Discloses real-world limitations and production considerations.

---

## 🛠️ Technology Stack

| Component | Technology | Purpose |
| :--- | :--- | :--- |
| **Language** | Python 3.10+ / 3.14 | Core application programming |
| **Web UI** | Streamlit | Reactive web dashboard & widgets |
| **ML Framework** | scikit-learn | Pipeline, TfidfVectorizer, MultinomialNB |
| **Data Handling** | pandas & numpy | Dataset loading, manipulation, and metric calculations |

---

## 📂 Project Structure

```text
AI SPAM EMAIL DETECTOR/
├── app.py              # Main Streamlit web application & ML pipeline
├── dataset.csv         # 100 realistic spam and legitimate demonstration samples
├── requirements.txt    # Project dependencies
└── README.md           # Project documentation and Viva guide
```

---

## 🚀 Getting Started on Windows

### 1. Prerequisites
Ensure Python is installed on your computer. You can verify this by opening **PowerShell** or **Command Prompt** and typing:
```bash
python --version
pip --version
```

### 2. Install Dependencies
Open PowerShell inside the project directory and run:
```powershell
pip install -r requirements.txt
```

### 3. Run the Application
Launch the Streamlit web server using Python's module launcher:
```powershell
python -m streamlit run app.py
```

### 4. Access the Dashboard
Once the command executes, Streamlit will automatically open your default web browser, or you can open:
```text
http://localhost:8501
```

---

## 📊 Machine Learning Pipeline Explanation

```text
Raw Text Input
      ↓
[ Preprocessing ]  --> Lowercasing, Stopword Removal, Alphanumeric Tokenization
      ↓
[ TF-IDF Vectorizer ] --> Converts text into Term Frequency-Inverse Document Frequency sparse vectors (n-grams: 1 to 2)
      ↓
[ Multinomial Naive Bayes ] --> Computes log-probabilities: P(Spam | Words) ∝ P(Spam) × ∏ P(Word_i | Spam)
      ↓
[ Decision Rule ]  --> Selects class with highest posterior probability: argmax(P(Class | Input))
```

### Mathematical Formulation for Viva:
1. **Term Frequency (TF)**:
   $$\text{TF}(t, d) = \frac{\text{Count of } t \text{ in } d}{\text{Total words in } d}$$
2. **Inverse Document Frequency (IDF)**:
   $$\text{IDF}(t) = \log\left(\frac{1 + N}{1 + \text{DF}(t)}\right) + 1$$
3. **Bayes Theorem for Naive Bayes**:
   $$P(\text{Spam} \mid X) = \frac{P(\text{Spam}) \prod_{i=1}^n P(x_i \mid \text{Spam})}{P(X)}$$
4. **Laplace Smoothing**:
   $$P(w_i \mid C) = \frac{N_{ci} + \alpha}{N_c + \alpha \cdot |V|}$$

---

## 🛡️ Academic Disclaimer
This system is an educational demonstration designed for college coursework, project evaluations, and viva presentations. A production enterprise spam filter (e.g., Gmail or Microsoft Defender) utilizes billions of training tokens, sender domain authentication (SPF, DKIM, DMARC), real-time DNS blocklists, and large-scale deep learning transformers.

---

## 👤 Author
**Developed by Prathmesh Khedkar**
