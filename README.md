# NLP Web App: Sentiment Analysis & POS Tagger

## 🧠 Project Overview
This is an interactive **web application** for **Sentiment Analysis** and **Part-of-Speech (POS) Tagging** built using **Streamlit**.  
It leverages **Hugging Face Transformers** models for accurate NLP tasks and provides **real-time visualization** for insights.  

---

## ⚡ Features
- **Sentiment Analysis**
  - Classifies text as **Positive** or **Negative**
  - Displays confidence scores and interactive charts
  - Highlights important words contributing to sentiment

- **POS Tagging**
  - Identifies **Nouns, Verbs, Adjectives, Adverbs, Pronouns, Determiners, Prepositions, Conjunctions**
  - Color-coded highlighting for each part of speech
  - Detailed table with confidence scores
  - Tag distribution visualized with **Pie and Bar charts**

- **Optimizations**
  - Models are **cached** for faster performance
  - **Eager model loading** for instant first-run analysis
  - Lexicon-based explanation via **VADER** (optional)

---

## 🛠️ Tech Stack
- **Python**  
- **Streamlit** – Web interface  
- **Hugging Face Transformers** – NLP models  
- **Torch** – Deep learning backend  
- **Plotly** – Interactive visualizations  
- **NLTK** – Lexicon-based sentiment analysis  
- **Pandas** – Data handling  

---

## 🚀 How to Run Locally

Clone the repository and run the project:

```bash
git clone https://github.com/&lt;your-username&gt;/&lt;repo-name&gt;.git
cd &lt;repo-name&gt;
python -m venv venv
.\venv\Scripts\Activate.ps1   
pip install -r requirements.txt
streamlit run app.py



⚠️ If PowerShell blocks the script, run:

Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

📝 Example Inputs & Outputs

Sentiment Analysis:

| Input Text                                     | Predicted Sentiment | Confidence |
| ---------------------------------------------- | ------------------- | ---------- |
| I love this product, it works amazingly well!  | Positive            | 98%        |
| This is the worst experience I've ever had.    | Negative            | 95%        |
| The movie was okay, not too bad but not great. | Negative            | 60%        |



POS Tagging:

| Input Text                                      | POS Highlights (Examples)                                                                  |
| ----------------------------------------------- | ------------------------------------------------------------------------------------------ |
| The quick brown fox jumps over the lazy dog.    | The(Det), quick(Adj), brown(Adj), fox(Noun), jumps(Verb), over(Prep), lazy(Adj), dog(Noun) |
| She happily sang beautiful songs in the garden. | She(Pron), happily(Adv), sang(Verb), beautiful(Adj), songs(Noun), in(Prep), garden(Noun)   |
| Running fast, he quickly finished the race.     | Running(Verb), fast(Adv), he(Pron), quickly(Adv), finished(Verb), race(Noun)               |



📂 Repository Files

app.py – Main Streamlit application
requirements.txt – Python dependencies
README.md – Project description

👩‍💻 Author

Developed by CHENNU BHAVANA 
