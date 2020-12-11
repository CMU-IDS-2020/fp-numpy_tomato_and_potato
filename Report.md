# Final Project Report

## **What makes a good movie?**

**Project URL**: https://share.streamlit.io/cmu-ids-2020/fp-numpy_tomato_and_potato/main

**Github Repo**: https://github.com/CMU-IDS-2020/fp-numpy_tomato_and_potato

**Dataset URL**: https://www.kaggle.com/rounakbanik/the-movies-dataset

<!-- Short (~250 words) abstract of the concrete data science problem and how the solutions addresses the problem.
 -->

## Introduction

Everyone loves movies! Whenever you are happy or sad, movies are one of the best companies for us. Especially under the COVID-19 pandemic, enjoying a movie can make us chill and escape from the reality for a while. But when you are watching the movie, do you ever think of what factors make it successful? We are going to use the data science method to reveal what makes a good movie. We used the ratings from the real users as a measure and explored the relationship between it and various characteristics. With many proactive questions being asked, we displayed the plots with different factors and custom interactive components accordingly. 

First, we explore the correlations between factors using heatmap and discovered many fun facts related to the movies. Second, we showed the rating distribution among all the genres and languages via diagram and violin plot. The budget is also an important element for the production, which will likely affect the movie quality, and we used a scatter plot to display the result. The casts and crews are the key components of a successful movie too since they are not quantitative, we used a network graph to analyze the top connected actors in top movies. In addition, from the keywords we can dig some valuable information, we used word-cloud to present the frequency of words among all the movies. In the end, we also provide a powerful machine learning model trained by xgboost to predict the rating of a new movie! ;)

## Related Work

There are many related works prior analysing movie dataset. They usually analyzed the relationships between numerical data type, such as ratings, runtime and budgets. Few of them analyzed nominal data type, such as generes. In previous works, most results are shown as a table, histogram graph or scatter point graph and lack the interaction ability. We think add more different types of graphs and interactions can help readers to understand the dataset better. Besides, few of the previous work analysis the directors, actors and keywords which we think may have a big influence. So, in this project, we explore more factors which may influence the ratings of movies and we use more visualization techniques and interaction widgets to explore the dataset deeply.

## Methods

#### Correlation Heatmap
We use a correlation heatmap to show how two factors are correlated with each other. We allow users to select factors they want to explore by multiselect widgets. Then, we use pearsonr function provided by scipy library to compute the correlation between every two factors. The pearsonr function actually computes the correlation coefficient for every pair factors based on the dataset we provided. Finally, we use the heatmap function provided by seaborn library to draw the heatmap.

#### Wordcloud

We analyze the keyword factor by using wordcloud visualization method. This method enable us to have a clear view of important keywords we are interested in. The main issue is how to determine which keywords are more important than others. We analyzed this in two ways. First we use count of appearance as the importance, this helps us to know which keywords are more used than others. If a keyword is used more, it is likely that keyword are more attractive to audience. So, the appearence count is one of the measure of importance. The other measure is TF-IDF value of that keyword. The insight behind TF-IDF is if a word appers more often in one document, but hardly to be found in other documents, that word will be more important. In order to use TF-IDF, we select a set of movies we are interested in, such like high-rating movies and low-rating movies, and we compare the keywords inside that group against other movies. In this way, we are able to know which keywords are prefered in that group of movies which gives us more insighs in how keywords influence the movies' ratings.

#### Network Graph

Ploting relations between actors is not straightforward, because it has to compute the location of each nodes over the layout according to its connection counts. We used cooperation in a movie as an edge and the actor as a vertex with the help of networkx library. For the visualization part, the figure is divided into edges and nodes. By the plotly figure, we combine them together to present the whole network.

#### XGBoost

## Results

For the visualization part, we draw a correlation heatmap for all factors. For genres analysis, we draw a histogram graph and a violin graph. For languages analysis, we draw a histogram graph and a violin graph. For budgets analysis, we draw a scatter plot and a box graph. For runtime analysis, we also draw a scatter plot and a box graph. For directors and actors analysis, we draw a cluster graph, a bar graph and a box graph. For keywords, we draw three word cloud graphs.

We explored different factors and found it is hard to find a decisive relationship between ratings and other factors. For example, the great directors can direct some terrible movies, and there are also some great movies with very low budget. However, we observed some fun relationships from the analysis process. 

In the correlation analysis, we found that vote count has a high correlation with profit and budget and profit has a correlation with profit. This seems that if we use budget for advertising and make more audience, we can have a higher profit. Besides, it's interesting to find budget has little relationship with rating. When analysing genres, we found that horror movies are tend to have a lower rating. When analysing languages, we found that malayalam tends to have a higher rating. When analysing runtime, we found that longer movies tend to have a higher rating. When analysing casts, we found that top movies usually have similar directors and actors. When analysing keywords, we found "woman director" is mostly used among all movies. High-rating movies tend to use "concert", "miniseries". Low-rating movies tend to use "shark", "dinosaur", "erotic movie".

## Discussion

The audience can learn how different factors influence ratings of movies and the correlationship between different factors. They can see what actors acts in top movies and their relatoinships. They can also see what directors directed most many high-rating movies. They can also know what keywords are popular in high-rating movies and low-rating movies. We enable the audience to interact with our website to discorver new relationships by themselves. Besides we build a prediction model, which the audience can interact with to predict a movie's rating.
## Future Work

* Streamlit requires computation on every component even when moving the sliders. We could further optimize it for smoother performance.

* While movies have other ratings on additional platforms like TMDB, MovieLens, etc. It will be interesting to explore the difference between those ratings on specific movies.  

## References
[Network Graphs in Python](https://plotly.com/python/network-graphs/)

[Data Science: Analysis of Movies released in the cinema between 2000 and 2017](https://medium.com/datadriveninvestor/data-science-analysis-of-movies-released-in-the-cinema-between-2000-and-2017-b2d9e515d032)

[Project Report: IMDB 5000 Movie Dataset](http://rstudio-pubs-static.s3.amazonaws.com/342210_7c8d57cfdd784cf58dc077d3eb7a2ca3.html)

[Hollywood Movie Data Analysis](https://static1.squarespace.com/static/55bfa8e4e4b007976149574e/t/5b998f398a922d8eaecaefd2/1536790332004/investigate-dataset-movies.pdf)