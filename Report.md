# Final Project Report

What makes a good movie?

**Project URL**: TODO

Github Repo: https://github.com/CMU-IDS-2020/fp-numpy_tomato_and_potato

Dataset URL: https://www.kaggle.com/rounakbanik/the-movies-dataset

<!-- Short (~250 words) abstract of the concrete data science problem and how the solutions addresses the problem.
 -->

## Introduction

Everyone loves movies! Whenever you are happy or sad, movies are one of the best companies for us. Especially under the COVID-19 pandemic, enjoying a movie can make us chill and escape from the reality for a while. But when you are watching the movie, do you ever think of what factors make it successful? We are going to use the data science method to reveal what makes a good movie. We used the ratings from the real users as a measure and explored the relationship between it and various characteristics. With many proactive questions being asked, we displayed the plots with different factors and custom interactive components accordingly. 

First, we explore the correlations between factors using heatmap and discovered many fun facts related to the movies. Second, we showed the rating distribution among all the genres and languages via diagram and violin plot. The budget is also an important element for the production, which will likely affect the movie quality, and we used a scatter plot to display the result. The casts and crews are the key components of a successful movie too since they are not quantitative, we used a network graph to analyze the top connected actors in top movies. In addition, from the keywords we can dig some valuable information, we used word-cloud to present the frequency of words among all the movies. In the end, we also provide a powerful machine learning model trained by xgboost to predict the rating of a new movie! ;)

## Related Work

Network Graphs: [Network Graphs in Python](https://plotly.com/python/network-graphs/)
## Methods

#### Heatmap

#### Wordcloud

#### Network Graph

Ploting relations between actors is not straightforward, because it has to compute the location of each nodes over the layout according to its connection counts. We used cooperation in a movie as an edge and the actor as a vertex with the help of networkx library. For the visualization part, the figure is divided into edges and nodes. By the plotly figure, we combine them together to present the whole network.

#### XGBoost

## Results

Key Takeways:

## Discussion

## Future Work

* Streamlit requires computation on every component even when moving the sliders. We could further optimize it for smoother performance.

* While movies have other ratings on additional platforms like TMDB, MovieLens, etc. It will be interesting to explore the difference between those ratings on specific movies.  