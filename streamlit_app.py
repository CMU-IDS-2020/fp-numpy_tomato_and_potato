from collections import Counter
import itertools
import streamlit as st
import pandas as pd
import altair as alt
import pydeck as pdk
import networkx as nx
import numpy as np
import plotly.graph_objs as go

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

def get_cast_pair_counter_edges(df_ratings, topNmovie):
  cast_pairs = get_cast_pairs(df_ratings, topNmovie)

  cast_counter = Counter(tuple(sorted(tup)) for tup in cast_pairs)

  cast_keys = list(cast_counter.keys())
  cast_values = list(cast_counter.values())
  edges = [(cast_keys[i][0], cast_keys[i][1], cast_values[i]) for i in range(len(cast_counter))]
  return cast_pairs, cast_counter, edges

def get_cast_pairs(df_ratings, topNmovie):
  cast_pairs = []
  movie_list = df_ratings[df_ratings['count'] >= 1000].sort_values(by=['score'], ascending = False)['movieId'].to_list()
  for movie_id in movie_list[:topNmovie]:
    casts_in_movies = list(df_casts[df_casts['movie_id'] == movie_id]['cast_id'])
    tups = list(itertools.combinations(casts_in_movies[:3], 2))
    cast_pairs.extend(tups)
  return cast_pairs

def get_layout(layout_option, G):
  if layout_option == 'random':
    return nx.random_layout(G)
  elif layout_option == 'circular':
    return nx.circular_layout(G)
  elif layout_option == 'kamada_kawai':
    return nx.kamada_kawai_layout(G)
  elif layout_option == 'mulitbipartite':
    return nx.mulitbipartite_layout(G)
  else:
    return nx.random_layout(G)


st.title("Movie")



###### PART N
###### Cluster graph of the connection between top actors of top movies
###### And the Top directors Bar chart
###### Need: movie_id list to filter the data to display (Top N rated or Top N bestseller ?)
###### and storytelling

layout_list = ['kamada_kawai', 'circular', 'random', 'multibipartite']

st.markdown('We are going to analyze the connections between actors in top movies')

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

st.markdown('We are going to analyze directors of those top movies')


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

st.markdown('We are going to analyze movie ratings of those top connected actors')

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

st.plotly_chart(box_connection_fig)


