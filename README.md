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
  - **Multi-threading** used to preload models
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
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
python -m venv venv
.\venv\Scripts\Activate.ps1   # PowerShell
pip install -r requirements.txt
streamlit run app.py

⚠️ If PowerShell blocks the script, run:

Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

📝 Example Inputs & Outputs

Sentiment Analysis:

<h3>Sentiment Analysis</h3>
<table>
  <tr>
    <th>Input Text</th>
    <th>Predicted Sentiment</th>
    <th>Confidence</th>
  </tr>
  <tr>
    <td>I love this product, it works amazingly well!</td>
    <td>Positive</td>
    <td>98%</td>
  </tr>
  <tr>
    <td>This is the worst experience I've ever had.</td>
    <td>Negative</td>
    <td>95%</td>
  </tr>
  <tr>
    <td>The movie was okay, not too bad but not great.</td>
    <td>Negative</td>
    <td>60%</td>
  </tr>
</table>

<h3>POS Tagging</h3>
<table>
  <tr>
    <th>Input Text</th>
    <th>POS Highlights (Examples)</th>
  </tr>
  <tr>
    <td>The quick brown fox jumps over the lazy dog.</td>
    <td>The(Noun), quick(Adj), brown(Adj), fox(Noun), jumps(Verb), lazy(Adj), dog(Noun)</td>
  </tr>
  <tr>
    <td>She happily sang beautiful songs in the garden.</td>
    <td>She(Pron), happily(Adv), sang(Verb), beautiful(Adj), songs(Noun), garden(Noun)</td>
  </tr>
  <tr>
    <td>Running fast, he quickly finished the race.</td>
    <td>Running(Verb), fast(Adv), he(Pron), quickly(Adv), finished(Verb), race(Noun)</td>
  </tr>
</table>

📂 Repository Files

app.py – Main Streamlit application
requirements.txt – Python dependencies
README.md – Project description

👩‍💻 Author

Developed by CHENNU BHAVANA 