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
  df_casts = pd.read_csv(cast_path)
  cast_id_name = pd.read_csv(cast_name_path)
  cast_id_map = cast_id_name.set_index('cast_id')['name'].to_dict()
  return df_casts, cast_id_map

@st.cache
def load_director_data():
  director_path = 'data/director.csv'
  df_director = pd.read_csv(director_path)
  return df_director

def get_cast_pairs(topNmovie, topNactor):
  cast_pairs = []
  for movie_id in df_casts['movie_id'].unique()[:topNmovie]:
    casts_in_movies = list(df_casts[df_casts['movie_id'] == movie_id]['cast_id'])
    tups = list(itertools.combinations(casts_in_movies[:topNactor], 2))
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

layout_list = ['random', 'circular', 'kamada_kawai', 'multibipartite']

st.markdown('We are going to analyze the connections between actors in top movies')

topNmovie = st.slider('Top N movies', 10, 200, value=50)

topNactor = st.slider('Top N actors', 1, 5, value=3)

layout_option = st.selectbox('Network layout', layout_list)


df_casts, cast_id_map = load_cast_data()

cast_pairs = get_cast_pairs(topNmovie, topNactor)

cast_counter = Counter(tuple(sorted(tup)) for tup in cast_pairs)

cast_keys = list(cast_counter.keys())
cast_values = list(cast_counter.values())
edges = [(cast_keys[i][0], cast_keys[i][1], cast_values[i]) for i in range(len(cast_counter))]


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

## to substitute by top voted id?
topmovies = df_director['movie_id'].unique()[:directortopNmovie]

filtered_director = df_director[df_director['movie_id'].isin(topmovies)]

filtered_director = filtered_director.groupby('name').count()

filtered_director = filtered_director.sort_values(by=['movie_id'], ascending = False)

director_fig = go.Figure(go.Bar(
            x=filtered_director['movie_id'][:10][::-1],
            y=filtered_director.index[:10][::-1],
            orientation='h'))

st.plotly_chart(director_fig)

