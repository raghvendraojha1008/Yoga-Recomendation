import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load your files
# df_user = pd.read_csv('health_lifestyle_dataset.csv')
# df_yoga = pd.read_excel('Yoga Data.xlsx')

def get_yoga_recommendation(user_profile):
    # 1. Combine yoga text data for searching
    df_yoga['tags'] = (df_yoga['Benefits'] + " " + 
                       df_yoga['Targeted Mental Problems'] + " " + 
                       df_yoga['Targeted Physical Problems']).fillna('')
    
    # 2. Match user goal with yoga tags using TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english')
    yoga_matrix = vectorizer.fit_transform(df_yoga['tags'])
    user_vec = vectorizer.transform([user_profile['goal']])
    
    # 3. Calculate similarity and get top matches
    scores = cosine_similarity(user_vec, yoga_matrix)
    df_yoga['score'] = scores[0]
    
    # 4. SAFETY FILTER: Remove if BP is high and contraindication matches
    if user_profile['bp_status'] == "High":
        df_yoga = df_yoga[~df_yoga['Contraindications'].str.contains("High Blood Pressure", na=False)]
        
    return df_yoga.sort_values(by='score', ascending=False).head(5)
