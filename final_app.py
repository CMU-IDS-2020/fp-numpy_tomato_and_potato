import json
import itertools
from collections import Counter
import numpy as np
import pandas as pd
import pydeck as pdk
import seaborn as sns
import networkx as nx
import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import plotly.express as px
import plotly.graph_objs as go
import plotly.figure_factory as ff
import xgboost
from xgboost import cv
from xgboost import plot_importance
from sklearn.model_selection import train_test_split,cross_val_score,KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error

_genre_level = 4000
_language_level = 1000

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


@st.cache
def load_genre_data():
    with open("data/genre_movie.json", 'r') as f:
        new_genre = json.load(f)
    return new_genre


@st.cache
def load_language_data():
    with open("data/language_movie.json", 'r') as f:
        new_language = json.load(f)
    return new_language


@st.cache
def load_budget_data():
    with open("data/budget_movie.json", 'r') as f:
        new_budget = json.load(f)
    return new_budget


@st.cache
def load_runtime_data():
    with open("data/runtime_movie.json", 'r') as f:
        new_runtime = json.load(f)
    return new_runtime


@st.cache
def load_movie_imdb():
    movies_imdb = pd.read_csv('data/movie_rating_imdb.csv')
    movies_imdb['tmdbId'] = movies_imdb['tmdbId'].astype(int) 
    return movies_imdb.set_index("tmdbId")['averageRating'].to_dict()


@st.cache
def load_keyword_data():
    keyword_path = 'data/keyword_ratings.csv'
    return pd.read_csv(keyword_path)


@st.cache
def load_cast_data():
    cast_path = "data/cast.csv"
    cast_name_path = "data/cast_id_name.csv"
    rating_path = "data/movie_ratings.csv"
    df_ratings = pd.read_csv(rating_path)
    df_casts = pd.read_csv(cast_path)
    cast_id_name = pd.read_csv(cast_name_path)
    cast_id_map = cast_id_name.set_index('cast_id')['name'].to_dict()
    return df_casts, df_ratings, cast_id_map


@st.cache
def load_director_data():
    director_path = 'data/director.csv'
    df_director = pd.read_csv(director_path)
    return df_director


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

    
def transform_id_to_rating(selected_keys, data):
    return {key: [imdb_rating[int(movie_id)] for movie_id in data[key]] for key in selected_keys}


def filter_genre_by_level(genre_dict, threshold):
    filter_genres = []
    for genre, movie_list in genre_dict.items():
        if len(movie_list) > threshold:
            filter_genres.append(genre)
    
    return filter_genres


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

@st.cache
def get_cast_pair_counter_edges(df_ratings, topNmovie):
    cast_pairs = get_cast_pairs(df_ratings, topNmovie)

    cast_counter = Counter(tuple(sorted(tup)) for tup in cast_pairs)

    cast_keys = list(cast_counter.keys())
    cast_values = list(cast_counter.values())
    edges = [(cast_keys[i][0], cast_keys[i][1], cast_values[i]) for i in range(len(cast_counter))]
    return cast_pairs, cast_counter, edges

@st.cache
def get_cast_pairs(df_ratings, topNmovie):
    cast_pairs = []
    movie_list = df_ratings[df_ratings['count'] >= 1000].sort_values(by=['score'], ascending = False)['movieId'].to_list()
    for movie_id in movie_list[:topNmovie]:
        casts_in_movies = list(df_casts[df_casts['movie_id'] == movie_id]['cast_id'])
        tups = list(itertools.combinations(casts_in_movies[:3], 2))
        cast_pairs.extend(tups)
    return cast_pairs

@st.cache
def get_layout(layout_option, G):
    if layout_option == 'random':
        return nx.random_layout(G)
    elif layout_option == 'circular':
        return nx.circular_layout(G)
    elif layout_option == 'kamada_kawai':
        return nx.kamada_kawai_layout(G)
    elif layout_option == 'mulitpartite':
        return nx.multipartite_layout(G)
    else:
        return nx.random_layout(G)


st.title("What makes good movies?")

st.write("Every one loves movies! From time to time, whenever you are happy or sad, movies are one of the best companies for you and me. Sometimes you met good movies which made you cry, other times you ran into bad movies which you couldn't finish. Data scientists like movies too! Based on the public data of about 45000 movies, we present this report on analyzing what makes the movies most people like, and we even build a model to predict the rating at the end. Enjoy the report!")

###### Factor correlations
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
budget_list = ['profit','budget']
runtime_list = ['runtime']
popularity_list = ['vote_count']
release_year_list = ['release_year']

st.markdown("# Factor Correlations")
st.write("In this part, we will make an overall analysis of the correlations between each factors, and the correlation between factors and rating. The measurement of correlation we are using is pearson correlation coefficient. Here is a factor corrlation heatmap you can play with and explore the correlations between different factors. For the categorical factors with many categories (casts, directors, genres), we only choose the most popular ones as examples.")

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
if "season" in x_list:
    x_factors+=season_list
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
if "season" in y_list:
    y_factors+=season_list
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

st.write("Most of people believe that the budget, casts and directors should be strongly correlated to the rating of a movie. However, the data tells a different story. From the heatmap above, we can see that the rating is not strongly correlated to any of the factors. This shows that the rating of a movie is very complicated, and there's no decisive factor for a good movie. ")
st.write("Disregard the fact that the absolute value of correlation is low between each factors and the rating, we can still see which factors are making positive/negative contribution to the ratings of movies. Surprisingly, the budget and release year both have negative impact on ratings (movie with less budget would have higher rating, and the older movies tend to have higher ratings.) Which might because people are tend to be nostalgia. Reguarding different genres, comedy is making a negative impact, while crime is making a positive impact. Also, we notice that some actors and directors also have positive impact to the rating of the movies.")
st.write("Comparing to the ratings, the profit and popularity (number of vote) of movies are more strongly correlated to some of the factors which we believed would have a strong correlation with rating. For example, the budget, runtime, some famous directors/actors.")
st.write("There are also some very interesting correlations between each factors. Of course Hitchcock is positively correlated to thriller movies, Quentin is positively correelated to crime movies, and Spielberg is positively correlated to Morgan Freeman. One finding very interesting for me is that action movie is positively correlated to Summer, which I never noticed before, but seems to be true according to my memory.  Play with the heatmap above and see what interesting fact you can find😄!")


################################## factor ##################################

st.markdown("# Which genres are your favorites?")
st.write("By default, we provide the top 6 genres whose total movie number is the most. However, \
        there are 20 genres in total, we allow users to multi select the genres which they are interested in. \
        here, we mainly explore on the top 6 genres.")
        
genre_data = load_genre_data()
imdb_rating = load_movie_imdb()
genre_rating = transform_id_to_rating(genre_data.keys(), genre_data)
first_level_genres = filter_genre_by_level(genre_data, _genre_level)

selected_genres = st.multiselect('What genre are you interested in?', list(genre_data.keys()), default=first_level_genres)
hist_fig = go.Figure()
for genre in selected_genres:
    hist_fig.add_trace(go.Histogram(x = genre_rating[genre],
                       opacity=0.75, name = genre,
                       xbins = dict(start = 0.0, end = 10.0, size = 0.2)))
hist_fig.update_layout(barmode='stack', xaxis_title_text='Rating', yaxis_title_text='Number')
st.plotly_chart(hist_fig, use_container_width=True)
st.write("As you can see from the histgram, firstly, most of the genres are rated between 6 and 8, \
    and it seems like drama and comedy are the most in this range (6.0~8.0). However, this doesn't \
    mean that the drama and comedy are more likely to get higher score. To explore the genre's \
    factors deeper, we plot violin figure to visualize each genre's rating distribution.")

violin_fig = go.Figure()
for genre in selected_genres:
    violin_fig.add_trace(go.Violin(y=genre_rating[genre], box_visible = True, 
                    meanline_visible=True, name=genre))
st.plotly_chart(violin_fig)
st.write("According to violin plot, we can find that drama is still the highest rated genre, since its \
    distribution is higher than other genres. Also, the median and max score for drama are also the highest\
    . And comedy movies, althought it seems to be highly rated from the histgram, the actual distribution \
    implies that Comedy is not highly rated in fact. Another interesting fact is that the Horror movie's \
    rarting is the lowest. The Horror movie's median score is 5.4, far less than other genres and the overall \
    distribution is lower also.")

st.write("All in all, it's obvious that the genre factor truly have an impact on the movie rating, and Drama \
    are the highest rated, while horror movies are the lowest rated")


st.markdown("# Beyond English")
st.write("Similar to Genre, we also allow users to select the movie language by themselves, and we provide \
    English, French, Italian, German, Japanaese languages by default.")
language_data = load_language_data()
language_rating = transform_id_to_rating(language_data.keys(), language_data)
first_level_languages = filter_genre_by_level(language_data, _language_level)
selected_languages = st.multiselect('What language movies are you interested in?', 
                                    list(language_data.keys()), default=first_level_languages)
language_hist_fig = go.Figure()
for language in selected_languages:
    language_hist_fig.add_trace(go.Histogram(x = language_rating[language],
                                opacity = 0.75, name = language,
                                xbins = dict(start = 0.0, end = 10.0, size = 0.2)))
language_hist_fig.update_layout(barmode = 'stack', xaxis_title_text='Rating', yaxis_title_text='Number')
st.plotly_chart(language_hist_fig, use_container_width=True)

st.write("Obviously, the English language movie has the largest number, however, the its distribution is \
    slightly lower than others (left in the Histgram figure). Also, we find that most of the movies are \
    rated between 6.0 and 8.0, this is consistent with previous genre's result. If you remove the English \
    movies and keep the rest default movies, you can find an interesting pehnomenon, the Japaneses movies \
    seem to have higher rating compared with other movies. To reduce the movies number's factor, and pay more \
    attention on the distribution, we plot the violin figure also.")

language_violin_fig = go.Figure()
for language in selected_languages:
    language_violin_fig.add_trace(go.Violin(y=language_rating[language],  box_visible = True,
                            meanline_visible=True, name=language))
st.plotly_chart(language_violin_fig)

st.write("Firstly, the violin plot validates our guessing that Japaneses movies are normally higher rated. Both \
    the entire distribution and mathematical statistics (median, mean) imply it. Secondly, we find that although \
    English movies number is far higher than other language movies, the overall rating is actually lower than \
    others. The reason might be that this dataset are collected in the USA, hence the foreign language movies are \
    more likely to be high-rated, the low-rated language movies may not be imported in the U.S actually. Thirdly, \
    we can conclude that Italian movies don't perform well also, the overall distribution is lower and there are \
    more low-rating Italian movies.")

st.write("To conclude it, the orignal language of movies truly influence the rating of movies. Even if we do not \
    consider the English movies. We can still find that Japanese movies are the best and Italian movies are the worst.")

st.markdown("# Bigger Budgets?")
st.write("Both language and genre are discrete variables, now we may focus more on continous variables, budget. Budget \
    is an interesting variables, naturally, we may think that higher budget should obtain higher revenue, however, \
    the rating score might be not high. Since it might sacrifice some culture value to appeal more people.")
budget_data = load_budget_data()
budget_rating = []
for budget, movie_list in budget_data.items():
    for movie_id in movie_list:
        budget_rating.append((budget, imdb_rating[movie_id]))
budget_df = pd.DataFrame(budget_rating, columns=["budget", "rating"])

budget_figure = px.scatter(budget_df, x="budget", y="rating")
budget_figure.update_layout(xaxis_title_text='Budget', yaxis_title_text='Rating')
st.plotly_chart(budget_figure)
st.write("We first use scatter plot to visualize the rough distribution. The X-axis is budget and Y-axis is rating. \
    The figure implies that there exist some relation between budget and rating. For example, we find that low budget \
    movies distribute from 0.0 to 10.0, which means there are not only some good movies but also bad movies with low \
    budget. Then, with the budget increasing, we find that bad movies (with low-rating score) decrease a lot. This implies \
    that, high budget movies will not be low-rated at least. To validate my guessing, I divide the budget into multiple groups \
    0~50M dollars, 50M~100M dollars, 100M~200M dollars, >200M dollars. Then, we can see the distribution in different budget \
    group.")

budget_bucket_rating = {"<50M":[], "50M~100M":[], "100M~200M":[], ">200M":[]}
for budget, movie_list in budget_data.items():
    budget = float(budget)
    if budget < 50000000:
        budget_key = "<50M"
    elif budget < 100000000:
        budget_key = "50M~100M"
    elif budget < 200000000:
        budget_key = "100M~200M"
    else:
        budget_key = ">200M"
    for movie_id in movie_list:
        budget_bucket_rating[budget_key].append(imdb_rating[movie_id])

budget_bucket_figure = go.Figure()
for budget_key in budget_bucket_rating:
    budget_bucket_figure.add_trace(go.Box(y=budget_bucket_rating[budget_key], name=budget_key))
st.plotly_chart(budget_bucket_figure)
st.write("We use boxplot to visualize the distributions in different budget group. Based on the figure, it's true that high budget \
    movies's rating distribution is overall higher than other distritbutions. This implies that if you pay more, even if you cannot \
    get the higest rating, you will not obtain too bad result. Also, we find that the highest rated movies lie in 100M~200M group \
    this is consistent with our common sense, that the medium-budget movies can be high rated.")

st.write("Hence, we think that the budget will also influence the movie rating.")

st.markdown("# Long Movies? Short Movies?")
st.write("Runtime is similar to Budget factor, we will also use scatter plot to visualize and explore it.")
runtime_data = load_runtime_data()
runtime_rating = []
for runtime, movie_list in runtime_data.items():
    for movie_id in movie_list:
        runtime_rating.append((runtime, imdb_rating[movie_id]))
runtime_df = pd.DataFrame(runtime_rating, columns=["runtime", "rating"])
runtime_figure = px.scatter(runtime_df, x="runtime", y="rating")
runtime_figure.update_layout(xaxis_title_text='Runtime', yaxis_title_text='Rating')
st.plotly_chart(runtime_figure)
st.write("From the runtime scatter plot, we can conclude that the movies with runtime > 400 are usually high rated. I think  \
    it might because that these movies are documentary, hence they are rated high. Also, some short movies with runtime < 50 \
    are also high rated. These movies might be the micro-movies, which are invariably highly rated also. Overall, we conclude \
    that there exists an trending that the movies with higher runtime might get higher score. To validate our guessing, we plot \
    boxplot figure also.")

runtime_bucket_rating = {"<60 min":[], "60~90 min":[], "90~150 min":[],"150~240 min":[], ">240 min":[]}
for runtime, movie_list in runtime_data.items():
    runtime = float(runtime)
    if runtime < 60:
        runtime_key = "<60 min"
    elif runtime < 90:
        runtime_key = "60~90 min"
    elif runtime < 150:
        runtime_key = "90~150 min"
    elif runtime < 240:
        runtime_key = "150~240 min"
    else:
        runtime_key = ">240 min"
    for movie_id in movie_list:
        runtime_bucket_rating[runtime_key].append(imdb_rating[movie_id])

runtime_bucket_figure = go.Figure()
for runtime_key in runtime_bucket_rating:
    runtime_bucket_figure.add_trace(go.Box(y=runtime_bucket_rating[runtime_key], name=runtime_key))
st.plotly_chart(runtime_bucket_figure)
st.write("The boxplot figure validates our guess. The movies whose runtime are between 60 and 150 minutes performs the worst, \
    and the micro-movies (with runtime < 60 mins) performs better. From the second group (60~90) to the fifth group (>240 min), \
    the distribution becomes gradually higher.")
st.write("As a consequence, we can conclude that the runtime also make some difference on the final rating, longer movies tend \
    to get higher rating score, except for too-short movies, which also get high scores.")



###### PART N
###### Cluster graph of the connection between top actors of top movies
###### And the Top directors Bar chart
###### Need: movie_id list to filter the data to display (Top N rated or Top N bestseller ?)
###### and storytelling

layout_list = ['kamada_kawai', 'circular', 'random', 'multibipartite']

st.markdown('# Directors and actos are making great movies!')
st.write('An important factor that affects the quality of the movie is the actors. Great movies often come with great casts. We are going to explore the connections between actors in top movies.')

st.write('From the top 250 rated movies with more than 1000 voters, we can see some famous actors stand out in the network graph, like Arnold Schwarzenegger, Matt Damon, and Chris Evans.')

st.write('This indicates that the participation of great casts has been related to the success of the movie. Feel free to explore it with different top movies and different network layouts!')

top_movie_list = [250, 200, 150, 100, 50]

topNmovie = st.selectbox('Top N movies', top_movie_list)


# topNmovie = st.slider('Top N movies', 10, 200, value=50)


layout_option = st.selectbox('Network layout', layout_list)


df_casts, df_ratings, cast_id_map = load_cast_data()


cast_pairs, cast_counter, edges = get_cast_pair_counter_edges(df_ratings, topNmovie)


G = nx.Graph()
G.add_edges_from(cast_pairs)

nodes = list(G.nodes)
d = dict(nx.degree(G))
# print(nx.info(G))

pos = get_layout(layout_option, G)

Xv = [pos[k][0] for k in nodes]
Yv = [pos[k][1] for k in nodes]

Xedge = []
Yedge = []

connect_count = list(d.values())

for edge in edges:
    Xedge+=[pos[edge[0]][0],pos[edge[1]][0], None]
    Yedge+=[pos[edge[0]][1],pos[edge[1]][1], None] 

edge_trace = go.Scatter(x=Xedge,
               y=Yedge,
               mode='lines',
               line=dict(color='#888', width=0.5),
               hoverinfo='none'
               )

node_trace = go.Scatter(x=Xv,
               y=Yv,
               mode='markers',
               name='net',
               line_width=2,
               marker=dict(symbol='circle-dot',
                             size=10, 
               showscale=True,
               colorscale = 'viridis',
               reversescale = False,
                             
               colorbar=dict(title='Number of Connections'),
               ),

               text=[(cast_id_map[nodes[i]], connect_count[i]) for i in range(len(nodes))], 
               hoverinfo='text'
               )
    

node_adjacencies = []
node_text = []
for node, adjacencies in enumerate(G.adjacency()):
    node_adjacencies.append(len(adjacencies[1]))

node_trace.marker.color = node_adjacencies


cluster_layout=go.Layout(title= "Connections Between Top Actors in Top Movies",  
    font= dict(size=12),
    showlegend=False, 
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
)

cluster_data = [node_trace, edge_trace]
cluster_fig = go.Figure(data=cluster_data, layout=cluster_layout)

st.plotly_chart(cluster_fig)


st.write('In addition, movies\' success also attribute to the director, which is the key role of the whole production.')

st.write('Of the top 10000 movies, we could see that Alfred Hitchcock, known as the "Master of Suspense", stands out for directing 52 of them. We can also see some prestigious name like Woody Allen, Robert Altman ... for making successful films.')

df_director = load_director_data()

directortopNmovie = st.slider('directors of top N movies', 100, 50000, value=10000)

### Top directors
topmovies = df_director['movie_id'].unique()[:directortopNmovie]

filtered_director = df_director[df_director['movie_id'].isin(topmovies)]

filtered_director = filtered_director.groupby('name').count()

filtered_director = filtered_director.sort_values(by=['movie_id'], ascending = False)

director_fig = go.Figure(go.Bar(
            x=filtered_director['movie_id'][:10][::-1],
            y=filtered_director.index[:10][::-1],
            orientation='h'))

st.plotly_chart(director_fig)

### Box plot of movie by the top connected actors

d = cast_pair_dictionary()
sort_connection_list = sorted(d.items(), key = lambda kv:(kv[1], kv[0]))
top10connection = sort_connection_list[::-1][:10]

top10cast = [t[0] for t in top10connection]

df_connection = df_casts[df_casts['cast_id'].isin(top10cast)]
df_connection_rating = df_connection.merge(df_ratings, left_on='movie_id', right_on='movieId')

box_traces = []
for i in range(len(top10cast)):
    trace = {
            "type": 'box',
            "x": df_connection_rating[df_connection_rating['cast_id'] == top10cast[i]]['name'],
            "y": df_connection_rating[df_connection_rating['cast_id'] == top10cast[i]]['score'],
            "name": cast_id_map[top10cast[i]],

        }
    box_traces.append(trace)
        
box_connection_fig = {
    "data": box_traces,
    "layout" : {
        "title": "Average Ratings of Top Connected Actors",
            "xaxis" : dict(title = 'Actor', showticklabels=True),
            "yaxis" : dict(title = 'Rating')
    }
}

st.write('As for the specific movies for those top connected actors, we could see that their ratings are above average and not that perfect. It indicates that great movies often come with great casts, but the participation of great actors cannot guarantee success.')
st.plotly_chart(box_connection_fig)



######################## keywords ##################3######
def tfidf(kwdf, agg_kwdf, n_keyword, n_film):
    feature_kwdf = pd.merge(kwdf, agg_kwdf, how='left', on=['keyword'])
    feature_kwdf['feature'] = feature_kwdf['count_x'] / n_keyword * np.log(n_film / (100*feature_kwdf['count_y']))
    return dict(zip(feature_kwdf['keyword'],feature_kwdf['feature']))

def draw_wc(frequency):
    wc = WordCloud(background_color="white", max_words=50)
    wc = wc.generate_from_frequencies(frequency)
    return wc

@st.cache()
def get_agg_kwdf(keyword_df):
  agg_kwdf = keyword_df.groupby('keyword') \
           .agg(count=('rating', 'size'))\
           .reset_index()
  return agg_kwdf

@st.cache()
def get_n_keyword(keyword_df):
  return len(keyword_df['keyword'].unique())

@st.cache()
def get_n_film(keyword_df):
  return len(keyword_df['movieId'].unique())

@st.cache()
def draw_wc_all(keyword_df):
  agg_kwdf = get_agg_kwdf(keyword_df)
  kw2count_all =dict(zip(agg_kwdf['keyword'], agg_kwdf['count']))
  return draw_wc(kw2count_all)

@st.cache()
def draw_wc_high(keyword_df):
  agg_kwdf = get_agg_kwdf(keyword_df)
  n_keyword = get_n_keyword(keyword_df)
  n_film = get_n_film(keyword_df)
  low_kwdf = keyword_df[keyword_df['rating'] < 5.0]
  agg_low_kwdf = low_kwdf.groupby('keyword') \
                  .agg(count=('rating', 'size')) \
                  .reset_index()
  kw2feature_low = tfidf(agg_low_kwdf, agg_kwdf, n_keyword, n_film)
  return draw_wc(kw2feature_low)

@st.cache()
def draw_wc_low(keyword_df):
  agg_kwdf = get_agg_kwdf(keyword_df)
  n_keyword = get_n_keyword(keyword_df)
  n_film = get_n_film(keyword_df)
  high_kwdf = keyword_df[keyword_df['rating'] >= 8.0]
  agg_high_kwdf = high_kwdf.groupby('keyword') \
                  .agg(count=('rating', 'size')) \
                  .reset_index()
  kw2feature_high = tfidf(agg_high_kwdf, agg_kwdf, n_keyword, n_film)
  return draw_wc(kw2feature_high)

keyword_df = load_keyword_data()

st.write("# What are good keywords?")

st.write("In this part, we explore the keywords of different movies by using word cloud graph. Keywords are an important part of movies as search engine uses them to generate search results. Good keywords help movies to be found by viewers who are interested in related genres, subjects, etc. Besides, keywords is one of the ways to leave first impressions on viewers, usually keywords involve the most important and distinct things of a movie. So it's important for us to know more about the keywords and how they affect the ratings of movies.")

st.write("We use world cloud as the visualization tool as we explore the influence of keywords. We first examine what keywords are most common among all movies. This will give us an insight on what keywords are popular and what keywords are rare. In the following word cloud graph are the most popular keywords, where the size of word represents the frequency of that keyword used among all movies in our dataset.")

wc = draw_wc_all(keyword_df)
fig, ax = plt.subplots(figsize=(20,10))
ax.imshow(wc, interpolation='bilinear')
ax.set_axis_off()
st.pyplot(fig)

st.write("Here we can find that \"woman director\" and \"independent film\" are the most popular keywords among all the movies, which means many movies use these two keywords. It's interesting to find many movies use \"woman director\" in the keywords and it seems people are focused more on women's rights these days. Some less popular keywords are \"murder\", \"based on novel\", \"musical\", \"biography\", \"violence\" which show the popular contents in movies these years. It implies these contents are wide-accepped among audience and thus movies keep on selecting these as their contents.")

st.write("In the following graphs, we use another method to show most important keywords in different range of ratings. The keywords are more important if they appear mostly in this range of ratings rather than in other ratings. To achieve that, we use TF-IDF as the importance of each keyword. If the keyword appears only in this range of ratings, it will have the highest TF-IDF value compared to other keywords. We first show the important key words in high-rate movies, which have IMDB rating at least 8.0.")

wc = draw_wc_high(keyword_df)
fig, ax = plt.subplots(figsize=(20,10))
ax.imshow(wc, interpolation='bilinear')
ax.set_axis_off()
st.pyplot(fig)

st.write("We can see in the graph that high-rate movies have keywords \"consert\", \"miniseries\", \"malayalam\". This shows that movies concerend with these keywords are more likely to have higher ratings. We then show the key words in low-rate movies, which have IMDB rating lower than 5.0.")

wc = draw_wc_low(keyword_df)
fig, ax = plt.subplots(figsize=(20,10))
ax.imshow(wc, interpolation='bilinear')
ax.set_axis_off()
st.pyplot(fig)

st.write("From this graph, we can see that \"shark\", \"dinosaur\", \"erotic movie\" are more likely to apper in low-rate movies. Also there are some other keywords like \"horror\", \"alien invasion\". These keywords are fascinating to audience at the first place and we are often tempted to watch some of them. However, it is showed these movies are more likely to be low-rate movies and are not acceppted by the crowd. Though there are good horror or shark movies, more movies of these types are unpleasant to watch. Usually these movies use these keywords to lure audience to go into the cinema but the contents of these movies are bad.")



######################### prediction ###########################
st.markdown("# Let's Predict")
st.write("At the end of this narrative, we provide you a chance to build your own machine learning model, and predict the rating of a movie of your choice! You can choose the features you want to use from all the features we mentioned previously. We will train a xgboost model for you on 90% of the data, and evaluate the model on the rest of the data. So you can know how your model performs! We will also provide you an analysis on which factor contributes the most to the prediction. The factor contributes the most is very likely the factor which affect the movies' rating the most.")
prediction_df = one_hot_df.drop(['id','release_season','genres','cast','director','spoken_languages','profit','vote_count'],axis=1)

# customized data
st.write("Please select the feature you want to use to train the model:")

potential_features = ['budget', 'runtime', 'release_year','genre','cast','director','season']
customized_features = st.multiselect('features', potential_features, default = potential_features)
if len(customized_features)==0:
    st.write("Please select at least one feature to start the prediction")
else:
    customized_dataset = pick_feature(prediction_df,customized_features)
    st.write("The following is the importance of each features. The Y axis is the f score of the variable, which is the total number of time the model split on the factor. What factor is the most important one😊?")
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
        st.write("Now you have a chance to predict the rating of a new movie using the model you just trained! Be creative🎉, imagine Hitchcock directing a comedy, or Steven Spielberg working with Bruce Willis!")
        st.write("Please enter the data of the movie you want to predict:")
        input_list = {}
        for f in customized_features:
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
            #    st.write([a for a in list(dict_fscore) if a.startswith(f)])
                if f!="season":
                    notice = " (If none of the option applies, please leave it empty.)"
                    user_input = st.multiselect(f+notice,[a.split("_")[-1] for a in list(dict_fscore) if a.startswith(f)])
                    input_list[f] = user_input
                else:
                    user_input = st.selectbox(f,[a.split("_")[-1] for a in list(dict_fscore) if a.startswith(f)])
                    input_list[f] = user_input
    #    input_list={'budget':100000, 'runtime':80,'release_year':2002,'genre': ['Crime','Drama'],'season':'Fall','cast':['Al Pacino','Robert De Niro'],'director':['Martin Scorsese']}

    #    st.write(input_list)

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
        st.write("The estimated rating of this movie is: "+str(round(output[0],2))+" !")
