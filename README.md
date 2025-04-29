# 🎬 Movie Recommendation System

A content-based movie recommender built using [Sentence Transformers](https://www.sbert.net/) and cosine similarity. This system uses semantic embeddings of movie metadata to suggest similar titles.

---

## 📌 Features

- ✅ Semantic understanding of movie metadata (overview, genres, keywords, production companies)
- ✅ Embedding-based similarity with `all-MiniLM-L6-v2` (via SentenceTransformers)
- ✅ Lightweight Streamlit interface
- ✅ Clean modular codebase for preprocessing, embedding, and inference

---

## 🧠 How It Works

1. **Preprocessing**: Combines relevant movie metadata into a unified `tags` column (I had used [tmdb v11 dataset from kaggle](https://www.kaggle.com/datasets/asaniczka/tmdb-movies-dataset-2023-930k-movies)).
2. **Embedding Generation**: Uses SentenceTransformer (`all-MiniLM-L6-v2`) to generate vector representations.
3. **Similarity Calculation**: Computes cosine similarity between embeddings to find the most similar movies.
4. **User Interface**: Allows users to input a movie title and returns the top 10 similar ones.

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/Movie-Recommendation-System.git
cd Movie-Recommendation-System
python -m venv venv
# On Windows: .\venv\Scripts\activate
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2. Download dataset
1. Download the [Datasets.zip](https://drive.google.com/file/d/1NSKYH0Vv-Bux6uC-ah6pzOIgjaewxRZn/view?usp=sharing)
2. Extract it as per project structure shown below

---

## 📂 Project Structure
```
Movie-Recommendation-System/
├── main.py                      # Streamlit web app
├── DataPreprocessor.ipynb      # Notebook for cleaning and embedding data
├── requirements.txt            # Required libraries
├── README.md                   # Project documentation
└── Datasets/
    └── cleaned/
        ├── processed.csv         # Cleaned movie data
        ├── df.pkl              # Pickled dataframe with titles and tags
        └── embeddings.pkl      # Sentence BERT embeddings
```
---

## 🧪 Usage
```bash
streamlit run main.py
```
