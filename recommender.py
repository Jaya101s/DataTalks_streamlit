import os
import pandas as pd
from scipy.sparse import csr_matrix
import streamlit as st
from sklearn.neighbors import NearestNeighbors
import numpy as np
import joblib
from fuzzywuzzy import fuzz

data_path ="Data"
movies_filename = "movie.csv"
ratings_filename = "rating.csv"

@st.cache
def load_movie_data():
    df_movies = pd.read_csv(os.path.join(data_path,movies_filename),
    usecols = ['movieId','title'],
    dtype = {'movieId':'int32','title':'str'})
    return df_movies

@st.cache
def load_ratings_data(limit=2000000):
    df_ratings = pd.read_csv(os.path.join(data_path,ratings_filename),
    usecols = ['userId','movieId','rating'],
    dtype = {'userId':'int32','movieId':'int32','rating':'float32'})
    return df_ratings[:limit]

@st.cache
def get_movies_user_matrix(df_movies, df_ratings, popularity_thres = 50, ratings_thres = 50):
    df_movies_cnt = pd.DataFrame(df_ratings.groupby('movieId').size(), columns=['count'])
    popular_movies = list(set(df_movies_cnt.query('count >= @popularity_thres').index))
    df_ratings_drop_movies = df_ratings[df_ratings.movieId.isin(popular_movies)]
    df_users_cnt = pd.DataFrame(df_ratings_drop_movies.groupby('userId').size(), columns=['count'])
    active_users = list(set(df_users_cnt.query('count >= @ratings_thres').index))
    df_ratings_drop_users = df_ratings_drop_movies[df_ratings_drop_movies.userId.isin(active_users)]
    movie_user_mat = df_ratings_drop_users.pivot(index='movieId', columns='userId', values='rating').fillna(0)
    return movie_user_mat

def movie_to_idx(df_movies, movie_user_mat):
    movie_to_idx = { movie: i for i, movie in enumerate(list(df_movies.set_index('movieId').loc[movie_user_mat.index].title)) }
    return movie_to_idx

def get_movie_user_sparse_mat(movie_user_mat):
    movie_user_mat_sparse = csr_matrix(movie_user_mat.values)
    return movie_user_mat_sparse

def create_model(movie_user_mat_sparse):
    # num_users = len(df_ratings.userId.unique())
    # num_items = len(df_ratings.movieId.unique())
    # df_ratings_cnt_tmp = pd.DataFrame(df_ratings.groupby('rating').size(), columns=['count'])
    # total_cnt = num_users * num_items
    # rating_zero_cnt = total_cnt - df_ratings.shape[0]
    # df_ratings_cnt = df_ratings_cnt_tmp.append(pd.DataFrame({'count': rating_zero_cnt}, index=[0.0]),verify_integrity=True,).sort_index()
    # df_ratings_cnt['log_count'] = np.log(df_ratings_cnt['count'])
    model_knn = NearestNeighbors(metric='cosine',algorithm='brute',n_neighbors= 20, n_jobs=-1)
    model_knn.fit(movie_user_mat_sparse)
    return model_knn

def fuzzy_matching(mapper, fav_movie,):
    """
    return the closest match via fuzzy ratio. 
    
    Parameters
    ----------    
    mapper: dict, map movie title name to index of the movie in data

    fav_movie: str, name of user input movie
    
    Return
    ------
    index of the closest match
    """
    match_tuple = []
    # get match
    for title, idx in mapper.items():
        ratio = fuzz.ratio(title.lower(), fav_movie.lower())
        if ratio >= 60:
            match_tuple.append((title, idx, ratio))
    # sort
    match_tuple = sorted(match_tuple, key=lambda x: x[2])[::-1]
    matches = [x[0] for x in match_tuple]
    if not match_tuple:
        print('Oops! No match is found')
        return
    return match_tuple[0][1],matches


def make_recommendation(model_knn, data, mapper, fav_movie, n_recommendations):
    """
    return top n similar movie recommendations based on user's input movie


    Parameters
    ----------
    model_knn:  model,  model

    data: movie-user matrix

    mapper: dict, map movie title name to index of the movie in data

    fav_movie: str, name of user input movie

    n_recommendations: int, top n recommendations

    Return
    ------
    list of top n similar movie recommendations
    """
    model_knn.fit(data)
    results = [] 
    idx,matches = fuzzy_matching(mapper, fav_movie)
    if isinstance(idx,(int,float)):
        distances, indices = model_knn.kneighbors(data[idx], n_neighbors=n_recommendations+1)
        raw_recommends =  sorted(list(zip(indices.squeeze().tolist(), distances.squeeze().tolist())), key=lambda x: x[1])[:0:-1]
        reverse_mapper = {v: k for k, v in mapper.items()}
        for i, (idx, dist) in enumerate(raw_recommends):
            results.append({
                "movie":reverse_mapper[idx],
                "distance":dist
            })
        return matches,results
    else:
        print(idx, matches)

    


