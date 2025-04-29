import pickle
import streamlit as st
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load data
main_df = pickle.load(open('Datasets/cleaned/df.pkl', 'rb'))
embeddings = pickle.load(open('Datasets/cleaned/embeddings.pkl', 'rb'))
main_df['lower_title'] = main_df['title'].apply(lambda x: x.lower())

# Recommendation function
def recommend(movie):
    try:
        matches = main_df[main_df['lower_title'].str.contains(movie.lower())]
        if matches.empty:
            return None
        movie_index = matches.index[0]
        cosine_scores = cosine_similarity([embeddings[movie_index]], embeddings).flatten()
        similar_indices = np.argsort(cosine_scores)[::-1][1:11]
        return main_df.iloc[similar_indices][['title']]
    except Exception as e:
        return None

# Streamlit UI
st.title('🎬 Movie Recommendation System')

movie_name = st.text_input("Enter a movie name")

if st.button("Recommend"):
    titles = recommend(movie_name)
    if titles is None:
        st.error('No movies found with that name. Try exact or close match.')
    else:
        st.subheader("Top Recommendations:")
        for title in titles['title'].values:
            st.markdown(f"- {title}")
