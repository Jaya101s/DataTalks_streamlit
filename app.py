import os
import time
import random
import numpy as np
import pandas as pd
import numexpr as ne
import altair as alt
import seaborn as sns
from PIL import Image
import xgboost as xgb
import streamlit as st
from scipy import sparse
import plotly.express as px
from datetime import datetime
import matplotlib.pyplot as plt
from plotly import figure_factory as ff
from sklearn.metrics import mean_squared_error
from surprise.model_selection import GridSearchCV
from sklearn.metrics.pairwise import cosine_similarity
from surprise import SVD, SVDpp, KNNBaseline, BaselineOnly, Reader, Dataset
from project_utils import generate_csv,load_data, get_train_data,get_test_data,changingLabels
from project_utils import get_train_sample_sparse, get_test_sample_sparse, get_train_reg, get_test_reg
from project_utils import load_train_sparse_matrix, load_test_sparse_matrix, getAverageRatings, get_sample_sparse_matrix
from project_utils import make_table, error_metrics, plot_importance, train_test_xgboost, get_ratings, get_error,run_surprise
from project_utils import execute_train_test, get_surprise_base_model, get_surprise_knn_model, get_surprise_knn_item_model,get_xgb_bsl_knn,get_second_combo_model,get_combo_model,get_svdpp,get_matrix_factorization_svd
sns.set_style("whitegrid")


# VARIABLES
FIG_WIDTH = 12
FIG_HEIGHT = 8
netflix_rating_path = "Data/NetflixRatings.csv"
netflix_pkl = "Data/NetflixData.pkl"
train_reg_data_path = "Data/Train_Regression.csv"
test_reg_data_path = "Data/Test_Regression.csv"
my_seed = 15
random.seed(my_seed)
np.random.seed(my_seed)

# UI MENU
st.sidebar.image('images/ff.jpg',use_column_width=True)
st.sidebar.markdown('<h1 style="color:red">Datatalks</h1>',unsafe_allow_html=True)
st.sidebar.markdown("Data science project where **powerful and interactive graphics** will help you to precisely analyse streaming website datasets :sunglasses:")
page = st.sidebar.selectbox("Choose a page", ["Homepage","Visualization", "EDA", "AI","Try AI", "About"])
st.title(page)

# LOADING DATASET
st.subheader(":heavy_check_mark: status check 1")
data_load_state = st.text(':hourglass: Loading data...')
dataset = load_data(netflix_rating_path)
data_load_state.text('Loading data...done!')
train_data = get_train_data(dataset)
test_data = get_test_data(dataset)
TrainUISparseData = load_train_sparse_matrix(train_data)
TestUISparseData = load_test_sparse_matrix(test_data)

st.subheader(":heavy_check_mark: Creating sparse matrix for train Data")
rows,cols = TrainUISparseData.shape
presentElements = TrainUISparseData.count_nonzero()
st.write(f"Sparsity Of Train matrix : {(1-(presentElements/(rows*cols)))*100}% ")

st.subheader(":heavy_check_mark: Creating sparse matrix for test Data")
rows,cols = TestUISparseData.shape
presentElements = TestUISparseData.count_nonzero()
st.write(f"Sparsity Of Test matrix : {(1-(presentElements/(rows*cols)))*100}% ")

st.subheader("ML training dataset")
train_reg = get_train_reg(train_reg_data_path)
st.write(train_reg.head())

st.subheader("ML training dataset")
test_reg = get_test_reg(test_reg_data_path)
st.write(test_reg.head())

##################### HOMEPAGE ######################################################
if page.lower() == 'homepage':
    if not os.path.isfile(netflix_rating_path):
        generate_csv(netflix_rating_path)
        st.write(":hourglass: loading data from file as dataframe...")
    else:
        st.write(":heavy_check_mark: Data is available as dataframe")
    st.subheader(":heavy_check_mark: status check 2")
    if st.checkbox("Show dataset"):
        st.write(dataset)
        
    if st.checkbox("Show dataset details"):
        st.text("Show all the column Names")
        st.write(dataset.columns)
        st.text("Show column datatypes")
        st.write(dataset.dtypes)
        st.text("Show complete dataset size")
        st.write(dataset.shape)
        st.text("Show desc of Ratings in final data")
        st.write(dataset.describe())

    st.subheader(":heavy_check_mark: status check 3")
    st.write("Number of NaN values",dataset.isnull().sum())
    st.markdown("**conclusion** No null values were found.")

    st.subheader(":heavy_check_mark: status check 4")
    duplicates = dataset.duplicated(["MovieID","CustID", "Ratings"])
    st.write("Number of duplicate rows", str(duplicates.sum()))
    st.markdown("**conclusion** No duplicates were found.")

    st.subheader(":heavy_check_mark: Show total ratings, users & movies in dataset")
    st.write("Total number of movie ratings = ", str(dataset.shape[0]))
    st.write("Number of unique users = ", str(len(np.unique(dataset["CustID"]))))
    st.write("Number of unique movies = ", str(len(np.unique(dataset["MovieID"]))))

    st.subheader(":heavy_check_mark: splitting data into training & test datasets")

    st.text("train data 80%")
    st.write(train_data.head(10))
    st.text("test data 20%")
    st.write(test_data.head(10))

    st.header("select another page from sidebar")
##################### VISUALIZATION ############################################## 

elif page.lower() == 'visualization':
    st.header("train data 80%")
    st.area_chart(train_data)
    st.header("test data 80%")
    st.area_chart(test_data)

    st.header(":heavy_check_mark: Distribution of Ratings")
    plt.figure(figsize = (FIG_WIDTH, FIG_HEIGHT))
    ax = sns.countplot(x="Ratings", data=train_data)
    ax.set_yticklabels([changingLabels(num) for num in ax.get_yticks()])
    plt.tick_params(labelsize = 15)
    plt.title("Distribution of Ratings in train data", fontsize = 20)
    plt.xlabel("Ratings", fontsize = 20)
    plt.ylabel("Number of Ratings(Millions)", fontsize = 20)
    st.pyplot()
    st.write("This graph shows how **Distribution of Ratings** which shows the overall maturity level of the whole series and is provided by the audience ")

    st.header(":heavy_check_mark: Number of Ratings Per Month")
    train_data["DayOfWeek"] = train_data.Date.dt.weekday
    plt.figure(figsize = (FIG_WIDTH, FIG_HEIGHT))
    ax = train_data.resample("M", on = "Date")["Ratings"].count().plot()
    ax.set_yticklabels([changingLabels(num) for num in ax.get_yticks()])
    ax.set_title("Number of Ratings per Month", fontsize = 20)
    ax.set_xlabel("Date", fontsize = 20)
    ax.set_ylabel("Number of Ratings Per Month(Millions)", fontsize = 20)
    plt.tick_params(labelsize = 15)
    st.pyplot()
    st.write("This Graph will represents the **Number of Ratings Per Month** means counts of ratings grouped by months")

    st.header(":heavy_check_mark: Analysis of Ratings given by user")
    no_of_rated_movies_per_user = train_data.groupby(by = "CustID")["Ratings"].count().sort_values(ascending = False)
    fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize=(FIG_WIDTH, FIG_HEIGHT))
    sns.kdeplot(no_of_rated_movies_per_user.values, shade = True, ax = axes[0], bw=1.5)
    axes[0].set_title("Fig1", fontsize = 18)
    axes[0].set_xlabel("Number of Ratings by user", fontsize = 18)
    axes[0].tick_params(labelsize = 15)
    sns.kdeplot(no_of_rated_movies_per_user.values, shade = True, cumulative = True, ax = axes[1], bw=1.5)
    axes[1].set_title("Fig2", fontsize = 18)
    axes[1].set_xlabel("Number of Ratings by user", fontsize = 18)
    axes[1].tick_params(labelsize = 15)
    fig.subplots_adjust(wspace=2)
    plt.tight_layout()
    st.pyplot()
    st.write("Above graph shows that almost all of the users give very few ratings. There are very **few users who's ratings count is high** .Similarly, above fig2 graph shows that **almost 99% of users give very few ratings**")

    st.header(":heavy_check_mark: Analysis of Ratings Per User")
    quantiles = no_of_rated_movies_per_user.quantile(np.arange(0,1.01,0.01))
    fig = plt.figure(figsize = (FIG_WIDTH, FIG_HEIGHT))
    plt.title("Quantile values of Ratings Per User", fontsize = 20)
    plt.xlabel("Quantiles", fontsize = 20)
    plt.ylabel("Ratings Per User", fontsize = 20)
    plt.plot(quantiles)
    plt.scatter(x = quantiles.index[::5], y = quantiles.values[::5], c = "blue", s = 70, label="quantiles with 0.05 intervals")
    plt.scatter(x = quantiles.index[::25], y = quantiles.values[::25], c = "red", s = 70, label="quantiles with 0.25 intervals")
    plt.legend(loc='upper left', fontsize = 20)
    plt.tick_params(labelsize = 15)
    st.pyplot()
    st.write("This graph shows the Quantile values of Ratings Per User")

    st.header(":heavy_check_mark: Analysis of Ratings Per Movie")
    no_of_ratings_per_movie = train_data.groupby(by = "MovieID")["Ratings"].count().sort_values(ascending = False)
    fig = plt.figure(figsize = (FIG_WIDTH, FIG_HEIGHT))
    plt.title("Number of Ratings Per Movie", fontsize = 20)
    plt.xlabel("Movie", fontsize = 20)
    plt.ylabel("Count of Ratings", fontsize = 20)
    plt.plot(no_of_ratings_per_movie.values)
    plt.tick_params(labelsize = 15)
    st.pyplot()
    st.write("This graph shows the number of rating(in count) each movie achieved by the audience, which clearly shows that there are some movies which are very popular and were rated by many users as comapared to other movies ")

    st.header(":heavy_check_mark: Analysis of Movie Ratings on Day of Week")
    fig = plt.figure(figsize = (FIG_WIDTH, FIG_HEIGHT))
    axes = sns.countplot(x = "DayOfWeek", data = train_data)
    axes.set_title("Day of week VS Number of Ratings", fontsize = 20)
    axes.set_xlabel("Day of Week", fontsize = 20)
    axes.set_ylabel("Number of Ratings", fontsize = 20)
    axes.set_yticklabels([changingLabels(num) for num in ax.get_yticks()])
    axes.tick_params(labelsize = 15)
    st.pyplot()
    st.write("This graph will show Analysis of Movie Ratings on Day of Week in bar graph format ,here clearly visible that on sturday & sunday users are least interested in providing ratings ")

    st.header(":heavy_check_mark: 2nd Analysis of Movie Ratings on Day of Week")
    fig = plt.figure(figsize = (FIG_WIDTH, FIG_HEIGHT))
    sns.boxplot(x = "DayOfWeek", y = "Ratings", data = train_data)
    plt.title("Day of week VS Number of Ratings", fontsize = 20)
    plt.xlabel("Day of Week", fontsize = 20)
    plt.ylabel("Number of Ratings", fontsize = 20)
    plt.tick_params(labelsize = 15)
    st.pyplot()
    st.write("This graph will show Analysis of Movie Ratings on Day of Week in box plot format ,here clearly visible that on sturday & sunday users are least interested in providing ratings ")

    st.header(":heavy_check_mark: Analysis of Distribution of Movie ratings")
    average_ratings_dayofweek = train_data.groupby(by = "DayOfWeek")["Ratings"].mean()
    st.write("**Average Ratings on Day of Weeks**")
    st.write(average_ratings_dayofweek)

    st.header(":heavy_check_mark: This Average Ratings on Day of Weeks")
    st.area_chart(average_ratings_dayofweek)
    st.write("this graph represents that average rating is mostly lies between 3 to 4.")

    st.header(":heavy_check_mark: Distribution of Movie ratings amoung Users")
    plt.scatter(test_data["CustID"],test_data["MovieID"])
    st.pyplot()
    st.write("This Graph will show **Distribution of Movie ratings** amoung Users") 

##################### EDA ######################################################## 

elif page.lower() =='eda':
    st.header("Show unique customer & movieId in Train DataSet")
    st.write("Total number of movie ratings in train data = ", str(train_data.shape[0]))
    st.write("Number of unique users in train data = ", str(len(np.unique(train_data["CustID"]))))
    st.write("Number of unique movies in train data = ", str(len(np.unique(train_data["MovieID"]))))
    st.write("Highest value of a User ID = ", str(max(train_data["CustID"].values)))
    st.write("Highest value of a Movie ID =  ", str(max(train_data["MovieID"].values)))

    st.header("Show unique customer & movieId in Test DataSet")
    st.write("Total number of movie ratings in Test data = ", str(test_data.shape[0]))
    st.write("Number of unique users in Test data = ", str(len(np.unique(test_data["CustID"]))))
    st.write("Number of unique movies in trTestain data = ", str(len(np.unique(test_data["MovieID"]))))
    st.write("Highest value of a User ID = ", str(max(test_data["CustID"].values)))
    st.write("Highest value of a Movie ID =  ", str(max(test_data["MovieID"].values)))

    Global_Average_Rating = TrainUISparseData.sum()/TrainUISparseData.count_nonzero()
    st.write("Global Average Rating {}".format(Global_Average_Rating))

    AvgRatingUser = getAverageRatings(TrainUISparseData, True)
    Global_Average_Rating = TrainUISparseData.sum()/TrainUISparseData.count_nonzero()
    
    AvgRatingUser = getAverageRatings(TrainUISparseData, True)
    train_sample_sparse = get_train_sample_sparse(TrainUISparseData)
    test_sample_sparse = get_test_sample_sparse(TestUISparseData)
    
    globalAvgMovies = getAverageRatings(train_sample_sparse, False)
    globalAvgUsers = getAverageRatings(train_sample_sparse, True)
    sample_train_users, sample_train_movies, sample_train_ratings = sparse.find(train_sample_sparse)

    

##################### AI #########################################################

elif page.lower() =="ai":
    st.sidebar.title("Movie preferences :smile:")
    

    st.header("Transforming Data for Surprise Models")
    st.write(train_reg[['User_ID', 'Movie_ID', 'Rating']].head(5))
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(train_reg[['User_ID', 'Movie_ID', 'Rating']], reader)
    trainset = data.build_full_trainset()
    testset = list(zip(test_reg["User_ID"].values, test_reg["Movie_ID"].values, test_reg["Rating"].values))
    error_table = pd.DataFrame(columns = ["Model", "Train RMSE", "Train MAPE", "Test RMSE", "Test MAPE"])
    with st.spinner("Execute model creation"):
        model_train_evaluation, model_test_evaluation, error_table, fig = execute_train_test(train_reg, test_reg,error_table)
        st.pyplot(fig)
    st.balloons()
    st.write("Model Completed")

    #########  Surprise BaselineOnly Model ##########
    with st.spinner("Surprise BaselineOnly Model Running"):
        model_train_evaluation, model_test_evaluation, error_table, fig = get_surprise_base_model(trainset, testset, train_reg, test_reg, model_train_evaluation, model_test_evaluation, error_table)
        st.pyplot(fig)
    st.balloons()
    st.write("Baseline Model Completed")

    ########## Surprise KNN-Baseline with User-User ##########
    with st.spinner("Surprise KNN Model for User-User Similarity Running"):
        model_train_evaluation, model_test_evaluation, error_table = get_surprise_knn_model(data, trainset, testset, train_reg, test_reg, model_train_evaluation, model_test_evaluation,error_table)
    st.balloons()
    st.write("KNN Model for User-User Similarity Completed")

    ########## Surprise KNN-Baseline Item-Item Similarity ##########
    with st.spinner("Surprise KNN Model for Item-Item Similarity Running"):
        model_train_evaluation, model_test_evaluation, error_table = get_surprise_knn_item_model(data, trainset, testset,model_train_evaluation, model_test_evaluation,error_table)
    st.balloons()
    st.write("KNN Model for Item-Item Similarity Completed")


    st.header("Machine learning analysis data")
    st.write(error_table)

    train_reg["KNNBaseline_User"] = model_train_evaluation["KNNBaseline_User"]["Prediction"]
    train_reg["KNNBaseline_Item"] = model_train_evaluation["KNNBaseline_Item"]["Prediction"]

    test_reg["KNNBaseline_User"] = model_test_evaluation["KNNBaseline_User"]["Prediction"]
    test_reg["KNNBaseline_Item"] = model_test_evaluation["KNNBaseline_Item"]["Prediction"]
    st.write(test_reg.head())
    st.write("Number of nan values in Train Data "+str(train_reg.isnull().sum().sum()))
    
    ##########  ##########
    with st.spinner("Execute xgb_bsl_knn model creation"):
        model_train_evaluation, model_test_evaluation, error_table, fig = get_xgb_bsl_knn(train_reg, test_reg,model_train_evaluation, model_test_evaluation,error_table)
        st.pyplot(fig)
    st.balloons()
    st.write("xgb_bsl_knn Hybrid Model Completed")

##################### ABOUT ######################################################
elif page.lower() == "try ai":
    Movie = st.sidebar.multiselect("Which do you like the most?",("Avengers","The Golden Compass","Harry Potter"))
    director = st.sidebar.multiselect("Who is you fav director?",("Rv","Av","MP"))  
    genres = st.sidebar.multiselect("which topic you love",("Action","Horror","Thriller"))

elif page.lower() == "about":
    pass


##################### END ######################################################