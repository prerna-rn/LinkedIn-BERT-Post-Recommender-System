import warnings
import json
import streamlit as st
import requests
from streamlit_lottie import st_lottie
import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from transformers import RobertaTokenizer, RobertaModel
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import re
from streamlit.components.v1 import html
import spacy.cli
spacy.cli.download("en_core_web_sm")

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
st.set_option('deprecation.showPyplotGlobalUse', False)

# insert lottie animation
url = "https://assets1.lottiefiles.com/packages/lf20_ywt06tjx.json"
response = requests.get(url)

# Display the Lottie animation
st_lottie(response.json(), width=75, height=75)
st.title("LinkedIn Influencer Post Recommendations")


def main():
    st.markdown("Developed by Prerna Singh")
    # Title and description
    st.write("Get post recommendations, word cloud and similarity score.")
    # User input for name
    user_name = st.text_input("Enter a valid user name from the dataset:")
    if user_name:
        # Step 1: Collect data
        df = pd.read_csv('influencers_data_cleaned2.csv', engine="python")
        df = df.dropna()
        df = df.reset_index()

        # Step 2: Preprocess data
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')

        def preprocess(text):
            text = str(text).replace('\n', ' ').replace('\r', '')
            tokens = tokenizer.encode(text, add_special_tokens=True, max_length=512, truncation=True)
            input_ids = torch.tensor(tokens).unsqueeze(0)
            with torch.no_grad():
                outputs = model(input_ids)
                last_hidden_states = outputs[0][:, 0, :]
            return last_hidden_states.numpy().reshape(1, -1)

        post_embeddings = []
        batch_size = 32
        for i in range(0, len(df), batch_size):
            batch_embeddings = preprocess(df['content'][i:i + batch_size])
            post_embeddings.append(batch_embeddings)
        post_embeddings = np.concatenate(post_embeddings, axis=0)

        def recommend_posts(user_name, post_embeddings, n=10):
            user_posts = df[df['name'] == user_name]['content']
            user_post_embeddings = []
            for post in user_posts:
                user_post_embeddings.append(preprocess(post))
            user_post_embeddings = np.concatenate(user_post_embeddings, axis=0)
            user_profile_embedding = np.mean(user_post_embeddings, axis=0)
            similarity_scores = np.dot(post_embeddings, user_profile_embedding.T) / (
                        np.linalg.norm(post_embeddings, axis=1) * np.linalg.norm(user_profile_embedding))
            rankings = np.argsort(np.ravel(similarity_scores))[::-1]
            user_post_ids = df[df['name'] == user_name].index
            recommended_post_ids = np.delete(rankings, np.where(np.isin(rankings, user_post_ids)))
            recommended_posts = df.iloc[recommended_post_ids[:n]][['content']]
            recommended_embeddings = post_embeddings[recommended_post_ids[:n]]
            return recommended_posts, recommended_embeddings

        recommendations = recommend_posts(user_name, post_embeddings, n=10)
        recommended_post_content = recommendations[0]['content'].values.tolist()
        wordcloud_text = ' '.join(recommended_post_content)
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(wordcloud_text)

        model_name = 'roberta-base'
        tokenizer = RobertaTokenizer.from_pretrained(model_name)
        model = RobertaModel.from_pretrained(model_name)

        nlp = spacy.load("en_core_web_sm")

        def extract_interests(text):
            doc = nlp(text)
            interests = [re.sub(r'[,\n]', '', token.text) for token in doc if
                         token.text.lower() not in STOP_WORDS and len(token.text) > 1 and token.text.strip()]
            return interests

        user_interests = df[df['name'] == user_name]['content'].apply(extract_interests).explode().value_counts()
        user_interests = user_interests[user_interests.index != ''].index.str.strip().tolist()[:10]

        recommended_interests = recommendations[0]['content'].apply(extract_interests).explode().value_counts()
        recommended_interests = recommended_interests[recommended_interests.index != ''].index.str.strip().tolist()[:10]

        user_interests_encoded = tokenizer.batch_encode_plus(user_interests, padding=True, truncation=True,
                                                              return_tensors='pt')
        user_interest_embeddings = model(**user_interests_encoded)['last_hidden_state'].mean(dim=1).detach().numpy()

        recommended_interests_encoded = tokenizer.batch_encode_plus(recommended_interests, padding=True,
                                                                    truncation=True, return_tensors='pt')
        recommended_interest_embeddings = model(**recommended_interests_encoded)['last_hidden_state'].mean(dim=1).detach().numpy()

        similarities = cosine_similarity(user_interest_embeddings, recommended_interest_embeddings)
        overall_similarity = np.mean(similarities) * 100

        st.subheader(f"Recommended posts for {user_name}:")
        st.write(recommendations[0])

        st.subheader(f"Word Cloud of Recommended Posts for {user_name}:")
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f"Word Cloud of Recommended Posts for {user_name}")
        st.pyplot()

        st.subheader(f"Similarity Score:")

        st.write(f"Overall similarity between user interests and recommended interests: {overall_similarity:.2f}%")
                
if __name__ == '__main__':
    main()
