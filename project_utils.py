import os
import numpy as np
import pandas as pd
import xgboost as xgb
import streamlit as st
from scipy import sparse
from surprise import SVD
from surprise import SVDpp
import plotly.express as px
from joblib import dump,load
import matplotlib.pyplot as plt
from surprise import KNNBaseline
from surprise import BaselineOnly
from surprise import Reader, Dataset
from sklearn.metrics import mean_squared_error
from surprise.model_selection import GridSearchCV
from sklearn.metrics.pairwise import cosine_similarity

FIG_WIDTH = 12
FIG_HEIGHT = 8
MODEL_PATH = "ml_models/"

def generate_csv(netflix_rating_path):
    data = open(netflix_rating_path, mode = "w") 
    files = ['Data/combined_data_1.txt']
    for file in files:
        print("Reading from file: "+str(file)+"...")
        with open(file) as f:  
            for line in f:
                line = line.strip() 
                if line.endswith(":"):
                    movieID = line.replace(":", "")
                else:
                    row = [] 
                    row = [x for x in line.split(",")] #custID, rating and date are separated by comma
                    row.insert(0, movieID)
                    data.write(",".join(row))
                    data.write("\n")
        print("Reading of file: "+str(file)+" is completed\n")
    data.close()


@st.cache(allow_output_mutation=True)
def load_data(netflix_rating_path, colnames=["MovieID","CustID", "Ratings", "Date"]):
    data = pd.read_csv(netflix_rating_path, sep=",", names = colnames )
    data["Date"] = pd.to_datetime(data["Date"])
    data.sort_values(by = "Date", inplace = True)
    return data

@st.cache(allow_output_mutation=True)
def get_train_data(dataset):
    subset = dataset.iloc[:int(dataset.shape[0]*0.80)]
    subset.reset_index(drop=True, inplace=True)
    return subset

@st.cache(allow_output_mutation=True)
def get_test_data(dataset):
    subset = dataset.iloc[int(dataset.shape[0]*0.80):]
    subset.reset_index(drop=True, inplace=True)
    return subset

@st.cache(allow_output_mutation=True)
def get_train_reg(path):
    return pd.read_csv(path, names = ["User_ID", "Movie_ID", "Global_Average", "SUR1", "SUR2", "SUR3", "SUR4", "SUR5", "SMR1", "SMR2", "SMR3", "SMR4", "SMR5", "User_Average", "Movie_Average", "Rating"])

@st.cache(allow_output_mutation=True)
def get_test_reg(path):
    return pd.read_csv(path, names = ["User_ID", "Movie_ID", "Global_Average", "SUR1", "SUR2", "SUR3", "SUR4", "SUR5", "SMR1", "SMR2", "SMR3", "SMR4", "SMR5", "User_Average", "Movie_Average", "Rating"])

def changingLabels(number):
    return str(number/10**6) + "M"

def load_train_sparse_matrix(train_data):
    if os.path.isfile("Data/TrainUISparseData.npz"):
        TrainUISparseData = sparse.load_npz("Data/TrainUISparseData.npz")
    else:
        TrainUISparseData = sparse.csr_matrix((train_data.Ratings, (train_data.CustID, train_data.MovieID)))
        sparse.save_npz("Data/TrainUISparseData.npz", TrainUISparseData)
    return TrainUISparseData

def load_test_sparse_matrix(test_data):
    if os.path.isfile("Data/TestUISparseData.npz"):
        TestUISparseData = sparse.load_npz("Data/TestUISparseData.npz")
    else:
        TestUISparseData = sparse.csr_matrix((test_data.Ratings, (test_data.CustID, test_data.MovieID)))
        sparse.save_npz("Data/TestUISparseData.npz", TestUISparseData)
    return TestUISparseData

def getAverageRatings(sparseMatrix, if_user):
    ax = 1 if if_user else 0
    sumOfRatings = sparseMatrix.sum(axis = ax).A1  #this will give an array of sum of all the ratings of user if axis = 1 else 
    noOfRatings = (sparseMatrix!=0).sum(axis = ax).A1  #this will give a boolean True or False array, and True means 1 and False 
    rows, cols = sparseMatrix.shape
    averageRatings = {i: sumOfRatings[i]/noOfRatings[i] for i in range(rows if if_user else cols) if noOfRatings[i]!=0}
    return averageRatings

def get_sample_sparse_matrix(path,sparseMatrix, n_users, n_movies):
    users, movies, ratings = sparse.find(sparseMatrix)
    uniq_users = np.unique(users)
    uniq_movies = np.unique(movies)
    np.random.seed(15)   #this will give same random number everytime, without replacement
    userS = np.random.choice(uniq_users, n_users, replace = True)
    movieS = np.random.choice(uniq_movies, n_movies, replace = True)
    mask = np.logical_and(np.isin(users, userS), np.isin(movies, movieS))
    sparse_sample = sparse.csr_matrix((ratings[mask], (users[mask], movies[mask])),shape = (max(userS)+1, max(movieS)+1))
    sparse.save_npz(path, sparse_sample)
    return sparse_sample

def get_train_sample_sparse(TrainUISparseData, path = "Data/TrainUISparseData_Sample.npz"):
    ####Creating Sample Sparse Matrix for Train Data
    if not os.path.isfile(path):
        train_sample_sparse = get_sample_sparse_matrix(path,TrainUISparseData, 4000, 400)
    else:
        train_sample_sparse = sparse.load_npz(path)
    return train_sample_sparse

def get_test_sample_sparse(TestUISparseData, path = "Data/TestUISparseData_Sample.npz"):
    if not os.path.isfile(path):
        test_sample_sparse = get_sample_sparse_matrix(TestUISparseData, 2000, 200)
    else:
        test_sample_sparse = sparse.load_npz(path)
    return test_sample_sparse

def make_table(model_name, rmse_train, mape_train, rmse_test, mape_test,error_table):
    error_table = error_table.append(pd.DataFrame([[model_name, rmse_train, mape_train, rmse_test, mape_test]], columns = ["Model", "Train RMSE", "Train MAPE", "Test RMSE", "Test MAPE"]))
    error_table.reset_index(drop = True, inplace = True)
    return error_table

def error_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(abs((y_true - y_pred)/y_true))*100
    return rmse, mape

def plot_importance(model, clf):
    st.header("**This graph show the feature wise importance**")
    fig,ax = plt.subplots(figsize = (FIG_WIDTH, FIG_HEIGHT))
    model.plot_importance(clf, ax = ax, height = 0.3)
    plt.xlabel("F Score", )
    plt.ylabel("Features", )
    plt.title("Feature Importance", )
    return fig

def train_test_xgboost(x_train, x_test, y_train, y_test, model_name, error_table):
    train_result = dict()
    test_result = dict()   
    clf = xgb.XGBRegressor(n_estimators = 100, silent = False, n_jobs  = 10)
    clf.fit(x_train, y_train)
    y_pred_train = clf.predict(x_train)
    rmse_train, mape_train = error_metrics(y_train, y_pred_train)
    train_result = {"RMSE": rmse_train, "MAPE": mape_train, "Prediction": y_pred_train}
    y_pred_test = clf.predict(x_test)
    rmse_test, mape_test = error_metrics(y_test, y_pred_test)
    test_result = {"RMSE": rmse_test, "MAPE": mape_test, "Prediction": y_pred_test}
    fig = plot_importance(xgb, clf)
    error_table = make_table(model_name, rmse_train, mape_train, rmse_test, mape_test, error_table)
    save_model(clf, model_name=model_name, store_path= MODEL_PATH+f"{model_name}.dat")
    return train_result, test_result, error_table, fig

def get_ratings(predictions):
    actual = np.array([pred.r_ui for pred in predictions])
    predicted = np.array([pred.est for pred in predictions])
    return actual, predicted

def get_error(predictions):
    actual, predicted = get_ratings(predictions)
    rmse = np.sqrt(mean_squared_error(actual, predicted)) 
    mape = np.mean(abs((actual - predicted)/actual))*100
    return rmse, mape

def run_surprise(algo, trainset, testset, model_name, error_table):
    train = dict()
    test = dict()
    algo.fit(trainset)
    train_pred = algo.test(trainset.build_testset())
    train_actual, train_predicted = get_ratings(train_pred)
    train_rmse, train_mape = get_error(train_pred)
    train = {"RMSE": train_rmse, "MAPE": train_mape, "Prediction": train_predicted}
    test_pred = algo.test(testset)
    test_actual, test_predicted = get_ratings(test_pred)
    test_rmse, test_mape = get_error(test_pred)
    test = {"RMSE": test_rmse, "MAPE": test_mape, "Prediction": test_predicted}
    error_table = make_table(model_name, train_rmse, train_mape, test_rmse, test_mape, error_table)
    save_model(algo, model_name=model_name, store_path= MODEL_PATH+f"{model_name}.dat")
    return train, test ,error_table 

def execute_train_test(train_reg, test_reg, error_table, model_train_evaluation = dict(), model_test_evaluation= dict()):
    x_train = train_reg.drop(["User_ID", "Movie_ID", "Rating"], axis = 1)
    x_test = test_reg.drop(["User_ID", "Movie_ID", "Rating"], axis = 1)
    y_train = train_reg["Rating"]
    y_test = test_reg["Rating"]
    train_result, test_result, error_table, fig = train_test_xgboost(x_train, x_test, y_train, y_test, "XGBoost_13",error_table)
    model_train_evaluation["XGBoost_13"] = train_result
    model_test_evaluation["XGBoost_13"] = test_result
    return model_train_evaluation, model_test_evaluation, error_table, fig

def get_surprise_base_model(trainset,testset, train_reg, test_reg, model_train_evaluation,model_test_evaluation,error_table):
    bsl_options = {"method":"sgd", "learning_rate":0.01, "n_epochs":25}
    algo = BaselineOnly(bsl_options=bsl_options)
    train_result, test_result, error_table = run_surprise(algo, trainset, testset, "BaselineOnly", error_table)
    model_train_evaluation["BaselineOnly"] = train_result
    model_test_evaluation["BaselineOnly"] = test_result
    train_reg["BaselineOnly"] = model_train_evaluation["BaselineOnly"]["Prediction"]
    st.write("Number of nan values = "+str(train_reg.isnull().sum().sum()))
    test_reg["BaselineOnly"] = model_test_evaluation["BaselineOnly"]["Prediction"]
    test_reg.head()
    st.write("Number of nan values = "+str(test_reg.isnull().sum().sum()))
    x_train = train_reg.drop(["User_ID", "Movie_ID", "Rating"], axis = 1)
    x_test = test_reg.drop(["User_ID", "Movie_ID", "Rating"], axis = 1)
    y_train = train_reg["Rating"]
    y_test = test_reg["Rating"]
    train_result, test_result,error_table, fig = train_test_xgboost(x_train, x_test, y_train, y_test, "XGB_BSL",error_table)
    model_train_evaluation["XGB_BSL"] = train_result
    model_test_evaluation["XGB_BSL"] = test_result
    return model_test_evaluation, model_train_evaluation, error_table, fig


def get_surprise_knn_model(data, trainset,testset, train_reg, test_reg, model_train_evaluation,model_test_evaluation,error_table):
    param_grid  = {'sim_options':{'name': ["pearson_baseline"], "user_based": [True], "min_support": [2], "shrinkage": [60, 80, 80, 140]}, 'k': [5, 20, 40, 80]}
    gs = GridSearchCV(KNNBaseline, param_grid, measures=['rmse', 'mae'], cv=3)
    gs.fit(data)
    st.write("GRIDSEARCH best scores", gs.best_score['rmse'])
    st.write("GRIDSEARCH best parameters", gs.best_params['rmse'])
    sim_options = {'name':'pearson_baseline', 'user_based':True, 'min_support':2, 'shrinkage':gs.best_params['rmse']['sim_options']['shrinkage']}
    bsl_options = {'method': 'sgd'} 
    algo = KNNBaseline(k = gs.best_params['rmse']['k'], sim_options = sim_options, bsl_options=bsl_options)
    train_result, test_result, error_table = run_surprise(algo, trainset, testset, "KNNBaseline_User", error_table)
    model_train_evaluation["KNNBaseline_User"] = train_result
    model_test_evaluation["KNNBaseline_User"] = test_result
    return model_train_evaluation, model_test_evaluation, error_table

def get_surprise_knn_item_model(data, trainset, testset, model_train_evaluation, model_test_evaluation, error_table):
    param_grid  = {'sim_options':{'name': ["pearson_baseline"], "user_based": [False], "min_support": [2], "shrinkage": [60, 80, 80, 140]}, 'k': [5, 20, 40, 80]}
    gs = GridSearchCV(KNNBaseline, param_grid, measures=['rmse', 'mae'], cv=3)
    gs.fit(data)
    st.write("GRIDSEARCH best scores",gs.best_score['rmse'])
    st.write("GRIDSEARCH best parameters",gs.best_params['rmse'])
    sim_options = {'name':'pearson_baseline', 'user_based':False, 'min_support':2, 'shrinkage':gs.best_params['rmse']['sim_options']['shrinkage']}
    bsl_options = {'method': 'sgd'} 
    algo = KNNBaseline(k = gs.best_params['rmse']['k'], sim_options = sim_options, bsl_options=bsl_options)
    train_result, test_result,error_table = run_surprise(algo, trainset, testset, "KNNBaseline_Item",error_table)
    model_train_evaluation["KNNBaseline_Item"] = train_result
    model_test_evaluation["KNNBaseline_Item"] = test_result
    return model_train_evaluation, model_test_evaluation, error_table

def get_xgb_bsl_knn(train_reg,test_reg,model_train_evaluation, model_test_evaluation, error_table):
    x_train = train_reg.drop(["User_ID", "Movie_ID", "Rating"], axis = 1)
    x_test = test_reg.drop(["User_ID", "Movie_ID", "Rating"], axis = 1)
    y_train = train_reg["Rating"]
    y_test = test_reg["Rating"]
    train_result, test_result, error_table, fig = train_test_xgboost(x_train, x_test, y_train, y_test, "XGB_BSL_KNN",error_table)
    model_train_evaluation["XGB_BSL_KNN"] = train_result
    model_test_evaluation["XGB_BSL_KNN"] = test_result
    return model_train_evaluation,model_test_evaluation,error_table,fig


def get_matrix_factorization_svd(data, trainset,testset,model_train_evaluation,model_test_evaluation,error_table):
    param_grid  = {'n_factors': [5,7,10,15,20,25,35,50,70,90]}   #here, n_factors is the equivalent to dimension 'd' when matrix 'A'
    gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3)
    gs.fit(data)
    st.write("GRIDSEARCH best scores", gs.best_score['rmse'])
    st.write("GRIDSEARCH best parameters", gs.best_params['rmse'])
    algo = SVD(n_factors = gs.best_params['rmse']['n_factors'], biased=True, verbose=True)
    train_result, test_result,error_table = run_surprise(algo, trainset, testset, "SVD",error_table)
    model_train_evaluation["SVD"] = train_result
    model_test_evaluation["SVD"] = test_result
    return model_train_evaluation,model_test_evaluation,error_table

def get_svdpp(data, trainset, testset, model_train_evaluation, model_test_evaluation, error_table):
    param_grid = {'n_factors': [10, 30, 50, 80, 100], 'lr_all': [0.002, 0.006, 0.018, 0.054, 0.10]}
    gs = GridSearchCV(SVDpp, param_grid, measures=['rmse', 'mae'], cv=3)
    gs.fit(data)
    st.write("GRIDSEARCH best scores", gs.best_score['rmse'])
    st.write("GRIDSEARCH best parameters", gs.best_params['rmse'])
    algo = SVDpp(n_factors = gs.best_params['rmse']['n_factors'], lr_all = gs.best_params['rmse']["lr_all"], verbose=True)
    train_result, test_result,error_table = run_surprise(algo, trainset, testset, "SVDpp",error_table)
    model_train_evaluation["SVDpp"] = train_result
    model_test_evaluation["SVDpp"] = test_result
    return model_train_evaluation,model_test_evaluation,error_table

def get_combo_model(train_reg, test_reg, model_train_evaluation, model_test_evaluation, error_table):
    train_reg["SVD"] = model_train_evaluation["SVD"]["Prediction"]
    train_reg["SVDpp"] = model_train_evaluation["SVDpp"]["Prediction"]
    test_reg["SVD"] = model_test_evaluation["SVD"]["Prediction"]
    test_reg["SVDpp"] = model_test_evaluation["SVDpp"]["Prediction"]
    x_train = train_reg.drop(["User_ID", "Movie_ID", "Rating"], axis = 1)
    x_test = test_reg.drop(["User_ID", "Movie_ID", "Rating"], axis = 1)
    y_train = train_reg["Rating"]
    y_test = test_reg["Rating"]
    train_result, test_result,error_table,fig = train_test_xgboost(x_train, x_test, y_train, y_test, "XGB_BSL_KNN_MF",error_table)
    model_train_evaluation["XGB_BSL_KNN_MF"] = train_result
    model_test_evaluation["XGB_BSL_KNN_MF"] = test_result
    return model_train_evaluation,model_test_evaluation,error_table,fig

def get_second_combo_model(train_reg, test_reg, model_train_evaluation, model_test_evaluation, error_table):
    ############ Surprise KNN Baseline + SVD + SVDpp  ###############
    x_train = train_reg[["KNNBaseline_User", "KNNBaseline_Item", "SVD", "SVDpp"]]
    x_test = test_reg[["KNNBaseline_User", "KNNBaseline_Item", "SVD", "SVDpp"]]
    y_train = train_reg["Rating"]
    y_test = test_reg["Rating"]
    train_result, test_result,error_table,fig = train_test_xgboost(x_train, x_test, y_train, y_test, "XGB_KNN_MF",error_table)
    model_train_evaluation["XGB_KNN_MF"] = train_result
    model_test_evaluation["XGB_KNN_MF"] = test_result
    return  model_train_evaluation,model_test_evaluation,error_table,fig

def plot_model_evaluation(error_table):
    error_table2 = error_table.drop(["Train MAPE", "Test MAPE"], axis = 1)    
    fig = px.bar(data_frame=error_table2, x = "Model",y="Train RMSE",title="Train and Test RMSE and MAPE of all Models")
    return fig


def load_model(model_path):
    if os.path.exists(model_path):
        model = load(model_path)
        print("model loaded")
        return model
    else:
        print("model not found, please check path")
        return None

def save_model(model, model_name, store_path):
    if os.path.exists(store_path):
        print("overwriting model files")
    dump(model,store_path)
    print(f"saved {model_name} to {store_path}")

def delete_model(model_path):
    if os.path.exists(model_path):
        os.remove(model_path)
        print("deleted model")
    else:
        print("no model exists")

@st.cache
def get_movies():
    movie_titles = pd.read_csv('Data/movie_titles.csv', encoding = 'ISO-8859-1', header = None, names = ['Id', 'Year', 'Name']).set_index('Id')
    movie_titles.sort_values(by='Year',inplace=True)
    movie_titles.dropna(axis=0, inplace=True)
    movie_titles.Year = movie_titles.Year.astype(int)
    return movie_titles

@st.cache
def movie_summary(df):
    f = ['count','mean']
    df_movie_summary = df.groupby('MovieID')['Ratings'].agg(f)
    return df_movie_summary

def get_movie_by_ratings(customer_id,rating,dataset,df_title):
    cust_movies = dataset[(dataset['CustID'] == customer_id[0]) & (dataset['Ratings'] == rating)]
    cust_movies = cust_movies.set_index('MovieID')
    cust_movies = cust_movies.join(df_title)['Name']
    return cust_movies

@st.cache
def drop_movies(df_movie_summary,q = 0.7):
    df_movie_summary.index = df_movie_summary.index.map(int)
    movie_benchmark = round(df_movie_summary['count'].quantile(q),0)
    drop_movie_list = df_movie_summary[df_movie_summary['count'] < movie_benchmark].index
    return drop_movie_list

def recommend(movie_title, min_count,df_title,df_movie_summary,df_p):
    print("For movie ({})".format(movie_title))
    print("- Top 10 movies recommended based on Pearsons'R correlation - ")
    i = int(df_title.index[df_title['Name'] == movie_title][0])
    target = df_p[i]
    similar_to_target = df_p.corrwith(target)
    corr_target = pd.DataFrame(similar_to_target, columns = ['PearsonR'])
    corr_target.dropna(inplace = True)
    corr_target = corr_target.sort_values('PearsonR', ascending = False)
    corr_target.index = corr_target.index.map(int)
    corr_target = corr_target.join(df_title).join(df_movie_summary)[['PearsonR', 'Name', 'count', 'mean']]
    return corr_target[corr_target['count']>min_count][:10].to_string(index=False)

def recommend_enhanced(customer_id,algo,df_title,drop_movie_list,limit=10):
    cust_movies = df_title.copy()
    cust_movies = cust_movies.reset_index()
    cust_movies = cust_movies[~cust_movies['Id'].isin(drop_movie_list)]
    cust_movies['Estimate_Score'] = cust_movies['Id'].apply(lambda x: algo.predict(customer_id, x).est)
    cust_movies = cust_movies.drop('Id', axis = 1)
    cust_movies = cust_movies.sort_values('Estimate_Score', ascending=False)
    return cust_movies.head(limit)


def get_model_names():
    models = os.listdir(MODEL_PATH)
    modelist = [model.split('.')[0]  for model in models if 'dat' in model]
    return modelist[:5]
