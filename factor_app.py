import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
import json
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.express as px


_genre_level = 4000
_language_level = 1000

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
    
def transform_id_to_rating(selected_keys, data):
    return {key: [imdb_rating[int(movie_id)] for movie_id in data[key]] for key in selected_keys}


@st.cache()
def load_movie_imdb():
    movies_imdb = pd.read_csv('data/movie_rating_imdb.csv')
    movies_imdb['tmdbId'] = movies_imdb['tmdbId'].astype(int) 
    return movies_imdb.set_index("tmdbId")['averageRating'].to_dict()

def filter_genre_by_level(genre_dict, threshold):
    filter_genres = []
    for genre, movie_list in genre_dict.items():
        if len(movie_list) > threshold:
            filter_genres.append(genre)
    
    return filter_genres
st.markdown("# Explore the Genre's impact on the movie rating")
st.write("By default, we provide the top 6 genres whose total movie number is the most. However, \
        there are 20 genres in total, we allow users to multi select the genres which they are interested in. \
        here, we mainly explore on the top 6 genres.")
        
genre_data = load_genre_data()
imdb_rating = load_movie_imdb()
first_level_genres = filter_genre_by_level(genre_data, _genre_level)

selected_genres = st.multiselect('What genre are you interested in?', list(genre_data.keys()), default=first_level_genres)

@st.cache
def get_genre_rating(genre_data):
    return transform_id_to_rating(genre_data.keys(), genre_data)

@st.cache(suppress_st_warning=True)
def draw_hist_fig(selected_genres, genre_rating):
    st.write('draw hist fig')
    hist_fig = go.Figure()
    for genre in selected_genres:
        hist_fig.add_trace(go.Histogram(x = genre_rating[genre],
                        opacity=0.75, name = genre,
                        xbins = dict(start = 0.0, end = 10.0, size = 0.2)))
    hist_fig.update_layout(barmode='stack', xaxis_title_text='Rating', yaxis_title_text='Number')
    return hist_fig

genre_rating = get_genre_rating(genre_data)
hist_fig = draw_hist_fig(selected_genres, genre_rating)
st.plotly_chart(hist_fig, use_container_width=True)

st.write("As you can see from the histgram, firstly, most of the genres are rated between 6 and 8, \
    and it seems like drama and comedy are the most in this range (6.0~8.0). However, this doesn't \
    mean that the drama and comedy are more likely to get higher score. To explore the genre's \
    factors deeper, we plot violin figure to visualize each genre's rating distribution.")

@st.cache(suppress_st_warning=True)
def draw_violin_fig(selected_genres, genre_rating):
    st.write('draw violin fig')
    violin_fig = go.Figure()
    for genre in selected_genres:
        violin_fig.add_trace(go.Violin(y=genre_rating[genre], box_visible = True, 
    violin_fig.add_trace(go.Violin(y=genre_rating[genre], box_visible = True, 
        violin_fig.add_trace(go.Violin(y=genre_rating[genre], box_visible = True, 
                        meanline_visible=True, name=genre))
    return violin_fig
violin_fig = draw_violin_fig(selected_genres, genre_rating)
st.plotly_chart(violin_fig)

st.write("According to violin plot, we can find that drama is still the highest rated genre, since its \
    distribution is higher than other genres. Also, the median and max score for drama are also the highest\
    . And comedy movies, althought it seems to be highly rated from the histgram, the actual distribution \
    implies that Comedy is not highly rated in fact. Another interesting fact is that the Horror movie's \
    rarting is the lowest. The Horror movie's median score is 5.4, far less than other genres and the overall \
    distribution is lower also.")

st.write("All in all, it's obvious that the genre factor truly have an impact on the movie rating, and Drama \
    are the highest rated, while horror movies are the lowest rated")


st.markdown("# Explore the Movie Language's impact on the movie rating")
st.write("Similar to Genre, we also allow users to select the movie language by themselves, and we provide \
    English, French, Italian, German, Japanaese languages by default.")


@st.cache
def get_lang_rating(language_data):
    return transform_id_to_rating(language_data.keys(), language_data)

language_data = load_language_data()
language_rating = get_lang_rating(language_data)
first_level_languages = filter_genre_by_level(language_data, _language_level)
selected_languages = st.multiselect('What language movies are you interested in?', 
                                    list(language_data.keys()), default=first_level_languages)


@st.cache(suppress_st_warning=True)
def draw_lang_hist(selected_languages, language_rating):
    st.write('draw hist')
    language_hist_fig = go.Figure()
    for language in selected_languages:
        language_hist_fig.add_trace(go.Histogram(x = language_rating[language],
                                    opacity = 0.75, name = language,
                                    xbins = dict(start = 0.0, end = 10.0, size = 0.2)))
    language_hist_fig.update_layout(barmode = 'stack', xaxis_title_text='Rating', yaxis_title_text='Number')
    return language_hist_fig

language_hist_fig = draw_lang_hist(selected_languages, language_rating)
st.plotly_chart(language_hist_fig, use_container_width=True)

st.write("Obviously, the English language movie has the largest number, however, the its distribution is \
    slightly lower than others (left in the Histgram figure). Also, we find that most of the movies are \
    rated between 6.0 and 8.0, this is consistent with previous genre's result. If you remove the English \
    movies and keep the rest default movies, you can find an interesting pehnomenon, the Japaneses movies \
    seem to have higher rating compared with other movies. To reduce the movies number's factor, and pay more \
    attention on the distribution, we plot the violin figure also.")

@st.cache(suppress_st_warning=True)
def draw_lang_violin(selected_languages, language_rating):
    st.write('draw violin')
    language_violin_fig = go.Figure()
    for language in selected_languages:
        language_violin_fig.add_trace(go.Violin(y=language_rating[language],  box_visible = True,
                                meanline_visible=True, name=language))
    return language_violin_fig

language_violin_fig = draw_lang_violin(selected_languages, language_rating)
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

st.markdown("# Explore the Movie Budget's impact on the movie rating")
st.write("Both language and genre are discrete variables, now we may focus more on continous variables, budget. Budget \
    is an interesting variables, naturally, we may think that higher budget should obtain higher revenue, however, \
    the rating score might be not high. Since it might sacrifice some culture value to appeal more people.")
budget_data = load_budget_data()

@st.cache(suppress_st_warning=True)
def draw_budget_figure(budget_data, imdb_rating):
    st.write('draw')
    budget_rating = []
    for budget, movie_list in budget_data.items():
        for movie_id in movie_list:
            budget_rating.append((budget, imdb_rating[movie_id]))
    budget_df = pd.DataFrame(budget_rating, columns=["budget", "rating"])
    budget_figure = px.scatter(budget_df, x="budget", y="rating")
    budget_figure.update_layout(xaxis_title_text='Budget', yaxis_title_text='Rating')
    return budget_figure

budget_figure = draw_budget_figure(budget_data, imdb_rating)
st.plotly_chart(budget_figure)

st.write("We first use scatter plot to visualize the rough distribution. The X-axis is budget and Y-axis is rating. \
    The figure implies that there exist some relation between budget and rating. For example, we find that low budget \
    movies distribute from 0.0 to 10.0, which means there are not only some good movies but also bad movies with low \
    budget. Then, with the budget increasing, we find that bad movies (with low-rating score) decrease a lot. This implies \
    that, high budget movies will not be low-rated at least. To validate my guessing, I divide the budget into multiple groups \
    0~50M dollars, 50M~100M dollars, 100M~200M dollars, >200M dollars. Then, we can see the distribution in different budget \
    group.")

@st.cache(suppress_st_warning=True)
def draw_budget_bucket(budget_data, imdb_rating):
    st.write('draw')
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
    return budget_bucket_figure

budget_bucket_figure = draw_budget_bucket(budget_data, imdb_rating)
st.plotly_chart(budget_bucket_figure)
st.write("We use boxplot to visualize the distributions in different budget group. Based on the figure, it's true that high budget \
    movies's rating distribution is overall higher than other distritbutions. This implies that if you pay more, even if you cannot \
    get the higest rating, you will not obtain too bad result. Also, we find that the highest rated movies lie in 100M~200M group \
    this is consistent with our common sense, that the medium-budget movies can be high rated.")

st.write("Hence, we think that the budget will also influence the movie rating.")

st.markdown("# Explore the Movie Runtime's impact on the movie rating")
st.write("Runtime is similar to Budget factor, we will also use scatter plot to visualize and explore it.")
runtime_data = load_runtime_data()

@st.cache(suppress_st_warning=True)
def draw_runtime(runtime_data, imdb_rating):
    st.write('draw')
    runtime_rating = []
    for runtime, movie_list in runtime_data.items():
        for movie_id in movie_list:
            runtime_rating.append((runtime, imdb_rating[movie_id]))
    runtime_df = pd.DataFrame(runtime_rating, columns=["runtime", "rating"])
    runtime_figure = px.scatter(runtime_df, x="runtime", y="rating")
    runtime_figure.update_layout(xaxis_title_text='Runtime', yaxis_title_text='Rating')
    return runtime_figure

runtime_figure = draw_runtime(runtime_data, imdb_rating)
st.plotly_chart(runtime_figure)

st.write("From the runtime scatter plot, we can conclude that the movies with runtime > 400 are usually high rated. I think  \
    it might because that these movies are documentary, hence they are rated high. Also, some short movies with runtime < 50 \
    are also high rated. These movies might be the micro-movies, which are invariably highly rated also. Overall, we conclude \
    that there exists an trending that the movies with higher runtime might get higher score. To validate our guessing, we plot \
    boxplot figure also.")

@st.cache(suppress_st_warning=True)
def draw_runtime_bucket(runtime_data, imdb_rating):
    st.write('draw')
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
    return runtime_bucket_figure

runtime_bucket_figure = draw_runtime_bucket(runtime_data, imdb_rating)
st.plotly_chart(runtime_bucket_figure)
st.write("The boxplot figure validates our guess. The movies whose runtime are between 60 and 150 minutes performs the worst, \
    and the micro-movies (with runtime < 60 mins) performs better. From the second group (60~90) to the fifth group (>240 min), \
    the distribution becomes gradually higher.")
st.write("As a consequence, we can conclude that the runtime also make some difference on the final rating, longer movies tend \
    to get higher rating score, except for too-short movies, which also get high scores.")