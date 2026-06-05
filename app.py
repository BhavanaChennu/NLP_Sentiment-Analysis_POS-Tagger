import streamlit as st
from transformers import pipeline
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import re
import time

# Suppress transformers warnings
import warnings
warnings.filterwarnings("ignore", message="Some weights of the model checkpoint")
warnings.filterwarnings("ignore", message="This IS expected")
warnings.filterwarnings("ignore", message="This IS NOT expected")
warnings.filterwarnings("ignore", message="`return_all_scores` is now deprecated")

try:
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except Exception:
    VADER_AVAILABLE = False

st.set_page_config(page_title="Sentiment Analysis and Tagger", page_icon="🧠", layout="wide")

st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
    .big-text { font-size: 60px; text-align: center; color: #fff; }
    .result-text { font-size: 40px; font-weight: bold; text-align: center; color: #fff; }
    .score-text { font-size: 50px; font-weight: bold; text-align: center; color: #ff6b6b; }
    .highlight-pos { background-color: #b2f7b2; padding: 2px 6px; border-radius: 4px; color: #000; }
    .highlight-neg { background-color: #f7b2b2; padding: 2px 6px; border-radius: 4px; color: #000; }
    .emoji-box { display:flex; align-items:center; justify-content:center; height:220px; }
    .nav-title { text-align: center; color: #fff; font-size: 60px; margin-top: 50px; }
    .nav-subtitle { text-align: center; color: rgba(255,255,255,0.8); font-size: 24px; margin-bottom: 60px; }
    .nav-card {
        background: rgba(255,255,255,0.1);
        border-radius: 20px;
        padding: 40px;
        text-align: center;
        transition: all 0.3s ease;
        border: 2px solid rgba(255,255,255,0.2);
        cursor: pointer;
    }
    .nav-card:hover {
        transform: translateY(-10px);
        background: rgba(255,255,255,0.2);
        border-color: rgba(255,255,255,0.5);
        box-shadow: 0 20px 40px rgba(0,0,0,0.3);
    }
    .nav-icon { font-size: 80px; margin-bottom: 20px; }
    .nav-card-title { font-size: 28px; color: #fff; font-weight: bold; margin-bottom: 10px; }
    .nav-desc { font-size: 16px; color: rgba(255,255,255,0.8); }
    .loading-text { color: #fff; font-size: 20px; text-align: center; }
    .highlight-noun { background-color: #ffd93d; padding: 2px 6px; border-radius: 4px; color: #000; }
    .highlight-verb { background-color: #6bcb77; padding: 2px 6px; border-radius: 4px; color: #000; }
    .highlight-adj { background-color: #ff6b6b; padding: 2px 6px; border-radius: 4px; color: #fff; }
    .highlight-adv { background-color: #4d96ff; padding: 2px 6px; border-radius: 4px; color: #fff; }
    .highlight-pron { background-color: #9b59b6; padding: 2px 6px; border-radius: 4px; color: #fff; }
    .highlight-det { background-color: #e67e22; padding: 2px 6px; border-radius: 4px; color: #fff; }
    .highlight-prep { background-color: #1abc9c; padding: 2px 6px; border-radius: 4px; color: #fff; }
    .highlight-conj { background-color: #e74c3c; padding: 2px 6px; border-radius: 4px; color: #fff; }
</style>
""", unsafe_allow_html=True)

if 'current_page' not in st.session_state:
    st.session_state.current_page = "home"
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
if 'models_loading' not in st.session_state:
    st.session_state.models_loading = False

if 'sentiment_history' not in st.session_state:
    st.session_state.sentiment_history = []
if 'sentiment_result' not in st.session_state:
    st.session_state.sentiment_result = {'label': None, 'confidence': 0.0}
if 'sentiment_scores' not in st.session_state:
    st.session_state.sentiment_scores = {'POSITIVE': 0.0, 'NEGATIVE': 0.0}
if 'sentiment_input' not in st.session_state:
    st.session_state.sentiment_input = ""

if 'tag_history' not in st.session_state:
    st.session_state.tag_history = []
if 'tag_result' not in st.session_state:
    st.session_state.tag_result = None
if 'tag_input' not in st.session_state:
    st.session_state.tag_input = ""

@st.cache_resource
def load_sentiment_model():
    try:
        import torch
        device = 0 if torch.cuda.is_available() else -1
    except Exception:
        device = -1
    return pipeline("sentiment-analysis",
                    model="distilbert-base-uncased-finetuned-sst-2-english",
                    return_all_scores=True,
                    device=device)

@st.cache_resource
def load_pos_model():
    try:
        import torch
        device = 0 if torch.cuda.is_available() else -1
    except Exception:
        device = -1
    return pipeline("token-classification",
                    model="vblagoje/bert-english-uncased-finetuned-pos",
                    aggregation_strategy="simple",
                    device=device)

@st.cache_resource
def load_vader():
    if not VADER_AVAILABLE:
        return None
    try:
        import nltk
        try:
            nltk.data.find('sentiment/vader_lexicon.zip')
        except Exception:
            nltk.download('vader_lexicon', quiet=True)
        return SentimentIntensityAnalyzer()
    except Exception:
        return None

def warm_up_models():
    if st.session_state.models_loaded or st.session_state.models_loading:
        return
    st.session_state.models_loading = True
    try:
        sentiment_pipe = load_sentiment_model()
        _ = sentiment_pipe("test")
        pos_pipe = load_pos_model()
        _ = pos_pipe("test")
        _ = load_vader()
        st.session_state.models_loaded = True
        st.session_state.models_loading = False
    except Exception as e:
        st.session_state.models_loading = False
        st.error(f"Model loading failed: {e}")

@st.cache_data(ttl=3600, show_spinner=False)
def predict(text):
    classifier = load_sentiment_model()
    out = classifier(text)
    items = None
    if isinstance(out, list):
        first = out[0]
        if isinstance(first, list):
            items = first
        elif isinstance(first, dict):
            items = out
        else:
            raise TypeError(f"Unexpected model output inner type: {type(first)}")
    elif isinstance(out, dict):
        items = [out]
    else:
        raise TypeError(f"Unexpected model output type: {type(out)}")
    try:
        return {item["label"]: item["score"] for item in items}
    except Exception as e:
        raise TypeError(f"Unexpected item structure from model: {e}")

def explain_text(text):
    analyzer = load_vader()
    if not analyzer:
        return {"error": "VADER not available"}
    words = re.findall(r"\w[\w']*", text)
    contributions = []
    for w in words:
        score = analyzer.lexicon.get(w.lower())
        if score is not None:
            contributions.append((w, score))
    contributions.sort(key=lambda x: abs(x[1]), reverse=True)
    compound = analyzer.polarity_scores(text)['compound']
    return {"words": contributions, "compound": compound}

def get_pos_color(tag):
    tag = str(tag).upper()
    if any(x in tag for x in ['NOUN', 'NN', 'NNS', 'NNP', 'NNPS']):
        return 'highlight-noun'
    elif any(x in tag for x in ['VERB', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']):
        return 'highlight-verb'
    elif any(x in tag for x in ['ADJ', 'JJ', 'JJR', 'JJS']):
        return 'highlight-adj'
    elif any(x in tag for x in ['ADV', 'RB', 'RBR', 'RBS']):
        return 'highlight-adv'
    elif any(x in tag for x in ['PRON', 'PRP', 'PRP$']):
        return 'highlight-pron'
    elif any(x in tag for x in ['DET', 'DT']):
        return 'highlight-det'
    elif any(x in tag for x in ['ADP', 'IN']):
        return 'highlight-prep'
    elif any(x in tag for x in ['CONJ', 'CC']):
        return 'highlight-conj'
    else:
        return 'highlight-pos'

def format_pos_tag(tag):
    tag_map = {
        'NOUN': 'Noun', 'NN': 'Noun', 'NNS': 'Noun (Pl)', 'NNP': 'Proper Noun', 'NNPS': 'Proper Noun (Pl)',
        'VERB': 'Verb', 'VB': 'Verb', 'VBD': 'Verb (Past)', 'VBG': 'Verb (Gerund)', 'VBN': 'Verb (Participle)',
        'VBP': 'Verb', 'VBZ': 'Verb (3rd)',
        'ADJ': 'Adjective', 'JJ': 'Adjective', 'JJR': 'Adj (Comp)', 'JJS': 'Adj (Sup)',
        'ADV': 'Adverb', 'RB': 'Adverb', 'RBR': 'Adv (Comp)', 'RBS': 'Adv (Sup)',
        'PRON': 'Pronoun', 'PRP': 'Pronoun', 'PRP$': 'Possessive',
        'DET': 'Determiner', 'DT': 'Determiner',
        'ADP': 'Preposition', 'IN': 'Preposition',
        'CONJ': 'Conjunction', 'CC': 'Conjunction',
        'NUM': 'Number', 'CD': 'Cardinal Number',
        'PART': 'Particle', 'INTJ': 'Interjection'
    }
    return tag_map.get(str(tag).upper(), str(tag))

@st.cache_data(ttl=3600, show_spinner=False)
def predict_pos(text):
    pos_tagger = load_pos_model()
    results = pos_tagger(text)
    words = []
    current_word = ""
    current_tag = ""
    current_score = 0
    for item in results:
        word = item.get('word', '').replace('##', '')
        tag = item.get('entity_group', item.get('entity', 'X'))
        score = item.get('score', 0)
        if not word.startswith('##') and current_word:
            words.append({'word': current_word, 'tag': current_tag, 'score': current_score})
            current_word = word
            current_tag = tag
            current_score = score
        else:
            current_word += word.replace('##', '')
            if score > current_score:
                current_tag = tag
                current_score = score
    if current_word:
        words.append({'word': current_word, 'tag': current_tag, 'score': current_score})
    return words

def show_home():
    st.markdown('<h1 class="nav-title">🧠 Sentiment Analysis and Tagger</h1>', unsafe_allow_html=True)
    st.markdown('<p class="nav-subtitle">VibeChecker</p>', unsafe_allow_html=True)
    if st.session_state.models_loading:
        st.markdown('<p class="loading-text">⚡ Pre-loading models for instant analysis...</p>', unsafe_allow_html=True)
    elif st.session_state.models_loaded:
        st.markdown('<p class="loading-text" style="color: #90EE90;">✅ Models ready - instant analysis!</p>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="nav-card">
            <div class="nav-icon">😊</div>
            <div class="nav-card-title">Sentiment Analysis</div>
            <div class="nav-desc">Analyze the emotional tone of text using state-of-the-art transformer models with interactive visualizations.</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Go to Sentiment Analysis", use_container_width=True, key="btn_sentiment"):
            st.session_state.current_page = "sentiment"
            st.rerun()
    with col2:
        st.markdown("""
        <div class="nav-card">
            <div class="nav-icon">🏷️</div>
            <div class="nav-card-title">POS Tagging</div>
            <div class="nav-desc">Identify parts of speech (nouns, verbs, adjectives) with color-coded highlighting and detailed linguistic analysis.</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Go to POS Tagging", use_container_width=True, key="btn_tagging"):
            st.session_state.current_page = "tagging"
            st.rerun()

def show_sentiment():
    if st.button("← Back to Home", key="back_sentiment"):
        st.session_state.current_page = "home"
        st.rerun()
    st.title("🤖 Sentiment Analysis and Tagger")
    st.subheader("VibeChecker - Advanced Sentiment Analysis with Real-time Visualization")
    st.markdown("### ✨ Try an Example")
    examples = [
        "I love this product, it works amazingly well!",
        "This is the worst experience I've ever had.",
        "The movie was okay, not too bad but not great either."
    ]
    cols = st.columns(len(examples))
    for i, ex in enumerate(examples):
        if cols[i].button(ex, key=f"sent_ex_{i}"):
            st.session_state.sentiment_input = ex
            st.rerun()
    left, right = st.columns([2, 1])
    with left:
        text_input = st.text_area("Input Text", height=150,
                                  placeholder="Enter text to analyze...",
                                  key="sentiment_input")
        analyze_clicked = st.button("✨ Analyze", use_container_width=True, key="analyze_sentiment")
        if analyze_clicked and text_input:
            if not st.session_state.models_loaded:
                with st.spinner("Loading models (first time only)... This may take 10-20 seconds..."):
                    warm_up_models()
                    attempts = 0
                    while not st.session_state.models_loaded and attempts < 30:
                        time.sleep(0.5)
                        attempts += 1
            with st.spinner("Analyzing..."):
                start = time.time()
                try:
                    scores = predict(text_input)
                    elapsed = time.time() - start
                    positive_score = scores.get("POSITIVE", 0.0)
                    negative_score = scores.get("NEGATIVE", 0.0)
                except Exception as e:
                    elapsed = time.time() - start
                    st.error(f"Model error: {e}")
                    scores = {'POSITIVE': 0.0, 'NEGATIVE': 0.0}
                    positive_score = 0.0
                    negative_score = 0.0
                label = "POSITIVE" if positive_score > negative_score else "NEGATIVE"
                confidence = max(positive_score, negative_score)
                st.write(f"Prediction time: {elapsed:.2f}s")
                st.session_state.sentiment_history.insert(0, {
                    'text': text_input[:40] + "..." if len(text_input) > 40 else text_input,
                    'sentiment': label,
                    'confidence': confidence
                })
                st.session_state.sentiment_result = {
                    'label': label,
                    'confidence': confidence,
                    'elapsed': elapsed,
                    'text': text_input
                }
                st.session_state.sentiment_scores = {
                    'POSITIVE': positive_score,
                    'NEGATIVE': negative_score
                }
            st.markdown("### 🔍 Highlighted Text")
            words = text_input.split()
            explained = []
            for w in words:
                if st.session_state.sentiment_result['label'] == "POSITIVE":
                    explained.append(f"<span class='highlight-pos'>{w}</span>")
                else:
                    explained.append(f"<span class='highlight-neg'>{w}</span>")
            st.markdown(" ".join(explained), unsafe_allow_html=True)
            if scores.get('POSITIVE', 0) > 0 or scores.get('NEGATIVE', 0) > 0:
                st.markdown("### 📖 Explanation")
                exp = explain_text(text_input)
                if isinstance(exp, dict) and exp.get('error'):
                    st.info("Explanation not available (VADER lexicon missing).")
                else:
                    words = exp.get('words', [])
                    if words:
                        rows = []
                        for w, s in words[:8]:
                            cls = 'highlight-pos' if s > 0 else 'highlight-neg'
                            rows.append(f"<span class='{cls}'>{w}</span>: {s:+.2f}")
                        st.markdown("  \n".join(rows), unsafe_allow_html=True)
                    else:
                        st.markdown("No distinctive lexicon words found; the model used contextual signals.")
                    st.markdown(f"**VADER compound score:** {exp.get('compound', 0.0):+.2f}")
    with right:
        st.markdown("## 🧠 Result")
        lr = st.session_state.get('sentiment_result', {'label': None, 'confidence': 0.0})
        if lr.get('label') == "POSITIVE":
            emoji_html = '<div class="big-text">😊</div>'
            color = "#00cc00"
        elif lr.get('label') == "NEGATIVE":
            emoji_html = '<div class="big-text">😔</div>'
            color = "#ff6b6b"
        else:
            emoji_html = '<div class="big-text">🙂</div>'
            color = "#ffffff"
        st.markdown(f"<div class='emoji-box'>{emoji_html}</div>", unsafe_allow_html=True)
        if lr.get('label'):
            st.markdown(f'<div class="result-text" style="color: {color};">{lr["label"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="score-text">{lr["confidence"]:.1%}</div>', unsafe_allow_html=True)
            st.markdown("<p style='text-align: center;'>Confidence Score</p>", unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-text" style="color: #ffffff;">No result yet</div>', unsafe_allow_html=True)
            st.markdown('<div class="score-text">—</div>', unsafe_allow_html=True)
    if st.session_state.sentiment_result and st.session_state.sentiment_result.get('label'):
        st.markdown("---")
        st.subheader("📊 Analysis Results")
        col1, col2 = st.columns(2)
        lr = st.session_state.sentiment_result
        scores = st.session_state.sentiment_scores
        with col1:
            color = "#00cc00" if lr['label'] == "POSITIVE" else "#ff6b6b"
            try:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=round(lr['confidence'] * 100, 1),
                    title={'text': "Confidence %", 'font': {'size': 24}},
                    gauge={'axis': {'range': [0, 100]},
                           'bar': {'color': color},
                           'bgcolor': "lightgray"}
                ))
                fig.update_layout(height=400, font_size=16)
                st.plotly_chart(fig, use_container_width=True, key='gauge_chart')
            except Exception as e:
                st.error(f"Chart error: {e}")
        with col2:
            try:
                df = pd.DataFrame({'Type': ['Positive', 'Negative'],
                                   'Score': [scores.get('POSITIVE', 0.0), scores.get('NEGATIVE', 0.0)]})
                fig2 = px.bar(df, x='Type', y='Score', color='Type',
                             color_discrete_map={'Positive': '#00cc00', 'Negative': '#ff6b6b'},
                             text_auto='.0%')
                fig2.update_layout(height=400, font_size=16, showlegend=False)
                st.plotly_chart(fig2, use_container_width=True, key='bar_chart')
            except Exception as e:
                st.error(f"Chart error: {e}")
    if st.session_state.sentiment_history:
        st.markdown("---")
        st.subheader("🕐 Recent History")
        for item in st.session_state.sentiment_history[:5]:
            icon = "🟢" if item['sentiment'] == "POSITIVE" else "🔴"
            st.write(f"{icon} **{item['sentiment']}** ({item['confidence']:.0%}) - {item['text']}")
        if st.button("Clear History", key="clear_hist_sentiment"):
            st.session_state.sentiment_history = []
            st.rerun()
    st.markdown("---")
    st.markdown("Developed by CHENNU BHAVANA | Powered by VibeChecker, Streamlit and Hugging Face Transformers")

def show_tagging():
    if st.button("← Back to Home", key="back_tagging"):
        st.session_state.current_page = "home"
        st.rerun()
    st.title("🏷️ Sentiment Analysis and Tagger")
    st.subheader("VibeChecker - Part-of-Speech Tagging with Real-time Visualization")
    st.markdown("### ✨ Try an Example")
    examples = [
        "The quick brown fox jumps over the lazy dog.",
        "She happily sang beautiful songs in the garden.",
        "Running fast, he quickly finished the race."
    ]
    cols = st.columns(len(examples))
    for i, ex in enumerate(examples):
        if cols[i].button(ex, key=f"tag_ex_{i}"):
            st.session_state.tag_input = ex
            st.rerun()
    left, right = st.columns([2, 1])
    with left:
        text_input = st.text_area("Input Text", height=150,
                                  placeholder="Enter text to tag...",
                                  key="tag_input")
        tag_clicked = st.button("🏷️ Tag Text", use_container_width=True, key="analyze_tagging")
        if tag_clicked and text_input:
            if not st.session_state.models_loaded:
                with st.spinner("Loading models (first time only)... This may take 10-20 seconds..."):
                    warm_up_models()
                    attempts = 0
                    while not st.session_state.models_loaded and attempts < 30:
                        time.sleep(0.5)
                        attempts += 1
            with st.spinner("Tagging..."):
                start = time.time()
                try:
                    tags = predict_pos(text_input)
                    elapsed = time.time() - start
                    tag_counts = {}
                    for t in tags:
                        tag = t['tag']
                        tag_counts[tag] = tag_counts.get(tag, 0) + 1
                    st.session_state.tag_history.insert(0, {
                        'text': text_input[:40] + "..." if len(text_input) > 40 else text_input,
                        'word_count': len(tags),
                        'tags': list(tag_counts.keys())[:3]
                    })
                    st.session_state.tag_result = {
                        'tags': tags,
                        'elapsed': elapsed,
                        'text': text_input,
                        'counts': tag_counts
                    }
                    st.rerun()
                except Exception as e:
                    st.error(f"Tagging error: {e}")
        if st.session_state.tag_result:
            st.markdown("### 🔍 Tagged Text")
            tags = st.session_state.tag_result['tags']
            highlighted = []
            for item in tags:
                word = item['word']
                tag = item['tag']
                color_class = get_pos_color(tag)
                highlighted.append(f"<span class='{color_class}' title='{format_pos_tag(tag)}'>{word}</span>")
            st.markdown(" ".join(highlighted), unsafe_allow_html=True)
            st.markdown("### 📖 Legend")
            legend_items = [
                ("Noun", "highlight-noun", "🟨"),
                ("Verb", "highlight-verb", "🟩"),
                ("Adjective", "highlight-adj", "🟥"),
                ("Adverb", "highlight-adv", "🟦"),
                ("Pronoun", "highlight-pron", "🟪"),
                ("Determiner", "highlight-det", "🟧"),
                ("Preposition", "highlight-prep", "🟢"),
                ("Conjunction", "highlight-conj", "🔴")
            ]
            legend_cols = st.columns(4)
            for i, (name, cls, emoji) in enumerate(legend_items):
                with legend_cols[i % 4]:
                    st.markdown(f"{emoji} <span class='{cls}' style='padding: 4px 8px; font-size: 12px;'>{name}</span>", unsafe_allow_html=True)
    with right:
        st.markdown("## 🧠 Result")
        lr = st.session_state.get('tag_result')
        if lr:
            st.markdown(f'<div class="big-text">🏷️</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="result-text" style="color: #fff;">{len(lr["tags"])} Words</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="score-text">{lr["elapsed"]:.2f}s</div>', unsafe_allow_html=True)
            st.markdown("<p style='text-align: center;'>Processing Time</p>", unsafe_allow_html=True)
        else:
            st.markdown('<div class="big-text">📝</div>', unsafe_allow_html=True)
            st.markdown('<div class="result-text" style="color: #ffffff;">No result yet</div>', unsafe_allow_html=True)
            st.markdown('<div class="score-text">—</div>', unsafe_allow_html=True)
    if st.session_state.tag_result:
        st.markdown("---")
        st.subheader("📊 Analysis Results")
        lr = st.session_state.tag_result
        counts = lr['counts']
        if counts:
            col1, col2 = st.columns(2)
            with col1:
                df_tags = pd.DataFrame({
                    'Tag': [format_pos_tag(k) for k in counts.keys()],
                    'Count': list(counts.values())
                })
                fig = px.pie(df_tags, values='Count', names='Tag',
                            title='Tag Distribution',
                            color_discrete_sequence=px.colors.qualitative.Pastel)
                fig.update_layout(height=400, font_size=14)
                st.plotly_chart(fig, use_container_width=True, key='pie_chart')
            with col2:
                fig2 = px.bar(df_tags, x='Tag', y='Count', color='Tag',
                             color_discrete_sequence=px.colors.qualitative.Pastel,
                             text='Count')
                fig2.update_layout(height=400, font_size=14, showlegend=False)
                st.plotly_chart(fig2, use_container_width=True, key='bar_chart_tags')
        st.markdown("### 📋 Detailed Tags")
        tag_df = pd.DataFrame([
            {
                'Word': t['word'],
                'Tag': format_pos_tag(t['tag']),
                'Confidence': f"{t['score']:.2%}",
                'Raw Tag': t['tag']
            } for t in lr['tags']
        ])
        st.dataframe(tag_df, use_container_width=True, hide_index=True)
    if st.session_state.tag_history:
        st.markdown("---")
        st.subheader("🕐 Recent History")
        for item in st.session_state.tag_history[:5]:
            tags_str = ", ".join([format_pos_tag(t) for t in item['tags']])
            st.write(f"🏷️ **{item['word_count']} words** ({tags_str}) - {item['text']}")
        if st.button("Clear History", key="clear_hist_tagging"):
            st.session_state.tag_history = []
            st.rerun()
    st.markdown("---")
    st.markdown("Developed by CHENNU BHAVANA | Powered by VibeChecker, Streamlit and Hugging Face Transformers")

if st.session_state.current_page == "home":
    show_home()
elif st.session_state.current_page == "sentiment":
    show_sentiment()
elif st.session_state.current_page == "tagging":
    show_tagging()