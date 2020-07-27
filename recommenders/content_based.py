"""

    Content-based filtering for item recommendation.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: You are required to extend this baseline algorithm to enable more
    efficient and accurate computation of recommendations.

    !! You must not change the name and signature (arguments) of the
    prediction function, `content_model` !!

    You must however change its contents (i.e. add your own content-based
    filtering algorithm), as well as altering/adding any other functions
    as part of your improvement.

    ---------------------------------------------------------------------

    Description: Provided within this file is a baseline content-based
    filtering algorithm for rating predictions on Movie data.

"""

# Script dependencies
import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Importing data
movies = pd.read_csv('resources/data/movies.csv', sep = ',',delimiter=',')
ratings = pd.read_csv('resources/data/ratings.csv')
movies.dropna(inplace=True)

def data_preprocessing(subset_size):
    """Prepare data for use within Content filtering algorithm.

    Parameters
    ----------
    subset_size : int
        Number of movies to use within the algorithm.

    Returns
    -------
    Pandas Dataframe
        Subset of movies selected for content-based filtering.

    """
    # Split genre data into individual words.
    movies['keyWords'] = movies['genres'].str.replace('|', ' ')
    # Subset of the data
    movies_subset = movies[:subset_size]
    return movies_subset

# !! DO NOT CHANGE THIS FUNCTION SIGNATURE !!
# You are, however, encouraged to change its content.
def content_model(movie_list,top_n=10):
    """Performs Content filtering based upon a list of movies supplied
       by the app user.

    Parameters
    ----------
    movie_list : list (str)
        Favorite movies chosen by the app user.
    top_n : type
        Number of top recommendations to return to the user.

    Returns
    -------
    list (str)
        Titles of the top-n movie recommendations to the user.

    """
    # Initializing the empty list of recommended movies
    recommended_movies = []
    data = data_preprocessing(27000)
    # Instantiating and generating the count matrix
    count_vec = TfidfVectorizer()
    count_matrix = count_vec.fit_transform(data['keyWords'])
    indices = pd.Series(data['title'])
    cosine_sim = cosine_similarity(count_matrix, count_matrix)
    # Getting the index of the movie that matches the title
    idx_1 = indices[indices == movie_list[0]].index[0]
    idx_2 = indices[indices == movie_list[1]].index[0]
    idx_3 = indices[indices == movie_list[2]].index[0]

    listing_list = []

    index_list = list([idx_1,idx_2,idx_3])

    for idx in index_list:
        try:
            rank = cosine_sim[idx]
            score_series = pd.Series(rank).sort_values(ascending = False)
            listing_list.append(score_series)
        except:
            pass

    print(listing_list)
    if len(listing_list)==1:
        listings = listing_list[0]

        recommended_movies = []
        # Appending the names of movies
        top_50_indexes = list(listings.iloc[1:50].index)
        # Removing chosen movies
        top_indexes = np.setdiff1d(top_50_indexes,[idx_1,idx_2,idx_3])
        for i in top_indexes[:top_n]:
            recommended_movies.append(list(movies['title'])[i])
        return recommended_movies

    elif len(listing_list)>1:
        print('I get here')
        listings = listing_list[0]
        for listing_entry in listing_list[1:]:
            listings.append(listing_entry)

        listings = listings.sort_values(ascending=True)

        recommended_movies = []
        # Appending the names of movies
        top_50_indexes = list(listings.iloc[1:50].index)
        # Removing chosen movies
        top_indexes = np.setdiff1d(top_50_indexes,[idx_1,idx_2,idx_3])
        for i in top_indexes[:top_n]:
            recommended_movies.append(list(movies['title'])[i])
        return recommended_movies


    else:
        recommended_movies = ['Interstellar (2014)','Django Unchained (2012)',
                     'Dark Knight Rises, The (2012)',
                     'Avengers, The (2012)',
                     'Guardians of the Galaxy (2014)',
                     'The Martian (2015)',
                     'Wolf of Wall Street, The (2013)',
                     'The Imitation Game (2014)',
                     'Deadpool (2016)',
                     'The Hunger Games (2012)']

        return recommended_movies
