from collections import Counter
from scipy.stats import pearsonr
import itertools
import streamlit as st
import pandas as pd
import altair as alt
import seaborn as sns
import pydeck as pdk
import networkx as nx
import numpy as np
import plotly.graph_objs as go
import matplotlib
import matplotlib.pyplot as plt
import xgboost
from xgboost import cv
from xgboost import plot_importance
from sklearn.model_selection import train_test_split,cross_val_score,KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error

@st.cache
def load_one_hot_data():
    one_hot_path = 'data/ml_data_with_pro.csv'
    df_one_hot = pd.read_csv(one_hot_path)
    return df_one_hot

@st.cache
def cast_pair_dictionary():
  cast_pairs = []
  for movie_id in df_casts['movie_id'].unique():
    casts_in_movies = list(df_casts[df_casts['movie_id'] == movie_id]['cast_id'])
    tups = list(itertools.combinations(casts_in_movies[:3], 2))
    cast_pairs.extend(tups)
  graph = nx.Graph()
  graph.add_edges_from(cast_pairs)
  d = dict(nx.degree(graph))
  return d

def get_cov_matrix(feature_list_1, feature_list_2,df):
    cov_matrix = []
    x_list = feature_list_1
    y_list = feature_list_2
    for x in x_list:
        curr_list = []
        for y in y_list:
            cor,_ = pearsonr(df[x],df[y])
            curr_list.append(round(cor,2))
        cov_matrix.append(curr_list)
    return cov_matrix

def display_cov_heatmap(feature_list_1, feature_list_2, df):
    cov_mat = get_cov_matrix(feature_list_1, feature_list_2, df)
    fig = plt.figure()
    if max(max(cov_mat))==1:
        ax = sns.heatmap(cov_mat, vmin=(min(min(cov_mat))+1)/2, vmax=-(min(min(cov_mat))+1)/2, linewidths=.5, annot=True, cmap="YlGnBu",
                         xticklabels=feature_list_2, yticklabels=feature_list_1)
        st.pyplot(fig)
    else:
        ax = sns.heatmap(cov_mat, linewidths=.5, annot=True, cmap="YlGnBu",
                         xticklabels=feature_list_2, yticklabels=feature_list_1, center=0)
        st.pyplot(fig)

def pick_feature(dataset, feature_list):
    new_feature=[]
    for f in feature_list:
        temp=[col for col in dataset if col.startswith(f)]
        new_feature.append(temp)
    new_feature=[j for i in new_feature for j in i]
    new_dataset = dataset[new_feature]
    return new_dataset

st.title("Movie")



###### Factor correlations
st.header("Factor Correlations")
st.write("In this part, we will make an overall analysis of the correlations between factors and rating.")
one_hot_df = load_one_hot_data()
one_hot_df = one_hot_df[one_hot_df.budget!=0]
season_list = ['season_Spring','season_Summer','season_Fall','season_Winter']
casts_list = ['cast_Bess Flowers','cast_Samuel L. Jackson',
              'cast_Robert De Niro','cast_Bruce Willis',
              'cast_John Hurt','cast_Michael Caine',
              'cast_Nicolas Cage','cast_Brad Pitt',
              'cast_Morgan Freeman','cast_Eric Roberts',
              'cast_Philip Ettington','cast_Johnny Depp',
              'cast_Armin Mueller-Stahl','cast_Tom Hanks']
director_list = ['director_Alfred Hitchcock',
                 'director_Steven Spielberg',
                 'director_François Truffaut',
                 'director_John Ford',
                 'director_Michael Curtiz',
                 'director_Otto Preminger',
                 'director_Tim Burton',
                 'director_Werner Herzog',
                 'director_Christopher Nolan',
                 'director_Quentin Tarantino',
                 'director_Ang Lee',
                 'director_David Fincher',
                 'director_James Cameron',
                 'director_Woody Allen']
genre_list = ['genre_Drama', 'genre_Comedy', 'genre_Thriller', 'genre_Romance', 'genre_Action', 'genre_Crime', 'genre_Horror']
rating_list = ['imdb_rating']
budget_list = ['budget','profit']
runtime_list = ['runtime']
popularity_list = ['vote_count']
release_year_list = ['release_year']

st.header("Factor Correlation Heatmap")
st.write("Please feel free to change the factors on x and y axis")
est_list = ["rating", "budget and profit", "runtime","release year","season","casts","directors","popularity","genre"]
default_features = ["rating", "budget and profit", "runtime","release year","popularity"]
x_list = st.multiselect('y axis', est_list, default = default_features)
x_factors = []
if "rating" in x_list:
    x_factors+=rating_list
if "budget and profit" in x_list:
    x_factors+=budget_list
if "runtime" in x_list:
    x_factors+=runtime_list
if "release year" in x_list:
    x_factors+=release_year_list
if "popularity" in x_list:
    x_factors+=popularity_list
if "genre" in x_list:
    x_factors+=genre_list
if "casts" in x_list:
    default_casts = ['cast_Brad Pitt', 'cast_Nicolas Cage', 'cast_Morgan Freeman']
    x_actors = st.multiselect('actors', casts_list, default = default_casts, key=0)
    x_factors+=x_actors
if "directors" in x_list:
    default_directors = ['director_Alfred Hitchcock', 'director_Steven Spielberg', 'director_Quentin Tarantino']
    x_directors = st.multiselect('directors', director_list, default = default_directors, key=1)
    x_factors+=x_directors
#st.write(x_factors)
y_list = st.multiselect('x axis', est_list, default = default_features)
y_factors = []
if "rating" in y_list:
    y_factors+=rating_list
if "budget and profit" in y_list:
    y_factors+=budget_list
if "runtime" in y_list:
    y_factors+=runtime_list
if "release year" in y_list:
    y_factors+=release_year_list
if "popularity" in y_list:
    y_factors+=popularity_list
if "genre" in y_list:
    y_factors+=genre_list
if "casts" in y_list:
    default_casts = ['cast_Brad Pitt', 'cast_Nicolas Cage', 'cast_Morgan Freeman']
    y_actors = st.multiselect('actors', casts_list, default = default_casts, key=2)
    y_factors+=y_actors
if "directors" in y_list:
    default_directors = ['director_Alfred Hitchcock', 'director_Steven Spielberg', 'director_Quentin Tarantino']
    y_directors = st.multiselect('directors', director_list, default = default_directors, key=3)
    y_factors+=y_directors
try:
    display_cov_heatmap(x_factors,y_factors,one_hot_df)
except:
    st.write("Please select at least one factor for each axis from boxes above")
#est_neigh = st.selectbox('Location', df_listing['neighbourhood_cleansed'].unique())
#est_bed = st.selectbox('Number of Bedroom', df_listing['bedrooms'].unique())
#st.write("And Y axis")

st.header("Movie Rating Prediction")
prediction_df = one_hot_df.drop(['id','release_season','genres','cast','director','spoken_languages','profit','vote_count'],axis=1)

# customized data
st.write("Please select the feature you want to use to train the model:")

potential_features = ['budget', 'runtime', 'release_year','genre','cast','director','season']
customized_features = st.multiselect('features', potential_features, default = potential_features)
if len(customized_features)==0:
    st.write("Please select at least one feature to start the prediction")
else:
    customized_dataset = pick_feature(prediction_df,customized_features)

    # train test split
    y = prediction_df['imdb_rating']
    x = customized_dataset
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    X = x_train.values
    y = y_train.values

    # k fold crossvalidation
    kf = KFold(n_splits=10)
    kf.get_n_splits(X)

    KFold(n_splits=10, random_state=None, shuffle=False)
    for train_index, valid_index in kf.split(X):
        #     print("TRAIN:", train_index, "VALID:", valid_index)
        X_train, X_valid = X[train_index], X[valid_index]
        y_train, y_valid = y[train_index], y[valid_index]

    # data preparation
    feature_names = x_train.columns
    dtrain = xgboost.DMatrix(X_train, label=y_train, feature_names=feature_names)
    dvalid = xgboost.DMatrix(X_valid, label=y_valid, feature_names=feature_names)
    dtest = xgboost.DMatrix(x_test.values, feature_names=feature_names)
    watchlist = [(dtrain, 'train'), (dvalid, 'valid')]

    xgb_pars = {'colsample_bytree':0.8, 'gamma':0, 'learning_rate':0.01, 'max_depth':5, 'min_child_weight':1.5,
                'n_estimators':10000, 'reg_alpha':0.75, 'reg_lambda':0.45, 'subsample':0.6,
                'seed':42, 'eval_metric': 'rmse', 'objective': 'reg:linear'}

    model = xgboost.train(xgb_pars, dtrain, 500, watchlist, early_stopping_rounds=250,maximize=False, verbose_eval=0)

    #predict_fig = plt.figure()
    #plot_importance(model,max_num_features=15)
    #st.pyplot(predict_fig)

    categorical_feature = ['genre','director','cast','season']
    other = ['budget','runtime','release_year']
    customized_ca = [i for i in customized_features if i in categorical_feature]
    customized_other = [i for i in customized_features if i in other]

    dict_fscore = model.get_fscore()
    dict_sum_score = {}
    for i in customized_ca:
        temp=[dict_fscore[a] for a in list(dict_fscore) if a.startswith(i)]
        dict_sum_score[i] = sum(temp)
    for j in customized_other:
        dict_sum_score[j] = dict_fscore[j]

    #plot
    importance_df = pd.DataFrame({'keys':list(dict_sum_score.keys()),'score':list(dict_sum_score.values())}).sort_values(['score'],ascending=False)
    predict_fig = plt.figure()
    #importance_df.reset_index(drop=True,inplace=True)
    #importance_df
    sns.barplot(x="keys",y="score",data=importance_df)
    plt.xticks(rotation=40)
    st.pyplot(predict_fig)

    y_predict = model.predict(dtest)
    st.write("The mean squared error of the prediction based on these features is: "+str(round(mean_squared_error(y_predict,y_test),2)))
    st.write("The mean absolute error of the prediction based on these features is: "+str(round(mean_absolute_error(y_predict,y_test),2)))

    if st.checkbox("I want to predict the rating for a new movie!"):
        st.write("Please enter the data of the movie you want to predict:")
        input_list = {}
        for f in other+categorical_feature:
            if f in other:
                user_input = ""
                if f=="budget":
                    user_input = st.text_input(f,100000)
                if f=="runtime":
                    user_input = st.text_input(f,120)
                if f=="release_year":
                    user_input = st.text_input(f,2020)
                if user_input!="":
                    input_list[f] = int(user_input)
            else:
#                st.write([a for a in list(dict_fscore) if a.startswith(f)])
                if f!="season":
                    notice = " (If none of the option applies, please leave it empty.)"
                    user_input = st.multiselect(f+notice,[a.split("_")[-1] for a in list(dict_fscore) if a.startswith(f)])
                    input_list[f] = user_input
                else:
                    user_input = st.selectbox(f,[a.split("_")[-1] for a in list(dict_fscore) if a.startswith(f)])
                    input_list[f] = user_input
#        input_list={'budget':100000, 'runtime':80,'release_year':2002,'genre': ['Crime','Drama'],'season':'Fall','cast':['Al Pacino','Robert De Niro'],'director':['Martin Scorsese']}

#        st.write(input_list)

        total_column_names = customized_dataset.columns

        def convert_data (input_list):
            mykeys = list(input_list.keys())
            for i in mykeys:
                if i in customized_ca:
                    temp = tuple(input_list[i])
                    for j in total_column_names:
                        if j.endswith(temp): input_list[j] = True
                        elif j.startswith(i): input_list[j] = False
                    del input_list[i]
            return pd.DataFrame([input_list])

        sample = convert_data(input_list)
        sample = sample[customized_dataset.columns]    # match feature names
        #prediction
        dsample = xgboost.DMatrix(sample.values, feature_names=sample.columns)
        output = model.predict(dsample)
        st.write("The estimated rating of this movie is: "+str(output[0])+" !")


