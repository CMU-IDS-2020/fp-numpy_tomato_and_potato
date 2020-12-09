import streamlit as st
import pandas as pd
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt

@st.cache
def load_keyword_data():
  keyword_path = 'data/keyword_ratings.csv'
  return pd.read_csv(keyword_path)

#################### keywords #####################
def tfidf(kwdf, agg_kwdf, n_keyword, n_film):
    feature_kwdf = pd.merge(kwdf, agg_kwdf, how='left', on=['keyword'])
    feature_kwdf['feature'] = feature_kwdf['count_x'] / n_keyword * np.log(n_film / (100*feature_kwdf['count_y']))
    return dict(zip(feature_kwdf['keyword'],feature_kwdf['feature']))

def draw_wc(frequency):
    wc = WordCloud(background_color="white", max_words=50)
    wc = wc.generate_from_frequencies(frequency)
    return wc

keyword_df = load_keyword_data()

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

@st.cache(suppress_st_warning=True)
def draw_wc_all(keyword_df):
  agg_kwdf = get_agg_kwdf(keyword_df)
  kw2count_all =dict(zip(agg_kwdf['keyword'], agg_kwdf['count']))
  return draw_wc(kw2count_all)

@st.cache(suppress_st_warning=True)
def draw_wc_high(keyword_df):
  st.write('rerun high')
  agg_kwdf = get_agg_kwdf(keyword_df)
  n_keyword = get_n_keyword(keyword_df)
  n_film = get_n_film(keyword_df)
  low_kwdf = keyword_df[keyword_df['rating'] < 5.0]
  agg_low_kwdf = low_kwdf.groupby('keyword') \
                  .agg(count=('rating', 'size')) \
                  .reset_index()
  kw2feature_low = tfidf(agg_low_kwdf, agg_kwdf, n_keyword, n_film)
  return draw_wc(kw2feature_low)

@st.cache(suppress_st_warning=True)
def draw_wc_low(keyword_df):
  st.write('rerun low')
  agg_kwdf = get_agg_kwdf(keyword_df)
  n_keyword = get_n_keyword(keyword_df)
  n_film = get_n_film(keyword_df)
  high_kwdf = keyword_df[keyword_df['rating'] >= 8.0]
  agg_high_kwdf = high_kwdf.groupby('keyword') \
                  .agg(count=('rating', 'size')) \
                  .reset_index()
  kw2feature_high = tfidf(agg_high_kwdf, agg_kwdf, n_keyword, n_film)
  return draw_wc(kw2feature_high)

# filtered_kwdf = keyword_df[(keyword_df['rating'] >= rating_low) & (keyword_df['rating'] <= rating_high)]
# agg_filtered_kwdf = filtered_kwdf.groupby('keyword') \
                # .agg(count=('rating', 'size')) \
                # .reset_index()
# count in filtered rating / count in all
# feature_kwdf = pd.merge(agg_filtered_kwdf, agg_kwdf, how='left', on=['keyword'])
# feature_kwdf['feature'] = feature_kwdf['count_x'] / n_keyword * np.log(n_film / (100*feature_kwdf['count_y']))
# dict for wordcloud use
# kw2feature = dict(zip(feature_kwdf['keyword'],feature_kwdf['feature']))
# kw2count = dict(zip(agg_filtered_kwdf['keyword'],agg_filtered_kwdf['count']))
# draw count
# draw_wc(kw2count)
# draw feature
# draw_wc(kw2feature)

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