from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#Initialize Flask
website = Flask(__name__)

#clustered dataset
df = pd.read_csv("clustered_movies.csv")

#TF-IDF Vectorizer
tfidf = TfidfVectorizer(stop_words = 'english', max_features = 500) #removes basic words that dont add meaning and limits vocob
tfidf_matrix = tfidf.fit_transform(df['overview']) #matrix

#website page
@website.route("/", methods = ["GET", "POST"]) #url
def index():
    query = "" #what the user typed
    results = [] #output of movies
    inputed_title = "" #empty right now

    #submit the movie
    if request.method == "POST": 
        query = request.form['title'].strip().lower() #lowercase and no whitespace
        matches = df[df['original_title'].str.lower().str.contains(query)] #find the movie

        #if there are matches
        if not matches.empty:
            movie = matches.iloc[0] #get first match
            inputed_title = movie['original_title']

            cluster = movie['cluster'] #cluster
            cluster_movies = df[(df['cluster'] == cluster) & (df['original_title'] != movie['original_title'])].copy() #movies in the cluster

            movie_vec = tfidf.transform([movie['overview']]) #numerical vector for selected movie
            cluster_vecs = tfidf.transform(cluster_movies['overview']) #numerical vector for clustered movies

            cluster_movies['similarity'] = cosine_similarity(movie_vec, cluster_vecs).flatten() #compares similarities
            recommendations = cluster_movies.sort_values(by = 'similarity', ascending = False).head(10) #top 10 similar
            results = recommendations[['original_title', 'overview']].values.tolist()
        if matches.empty:
            result = []
            message = "No movies found."
    return render_template("index.html", query = query, results = results, inputed_title = inputed_title) #converts into list
if __name__ == '__main__':
    website.run(debug = True) #displays them