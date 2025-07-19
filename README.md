# Binge-Trend
A recommendation website using Data Science, Machine Learning, and NLP to suggest movies, web series, anime, and related news. 

### What does it do?
Either type an imaginary plot to get the best match, or select a movie and get similar movies, **with percentage of every single genre present in it**.

### Website's Frontend User Interface:
![img](https://raw.githubusercontent.com/adityapradhan202/Binge-Trend/refs/heads/main/project_imgs/app_img.png)

### Working of recommendation system:
![working](https://raw.githubusercontent.com/adityapradhan202/Binge-Trend/refs/heads/main/project_imgs/fusinator_explaination.png)
Custom ensembled model is designed and trained so that we can overcome the problem of having very limited amount of data for training.
For searching similar movies, content based filtering has been used using the concept of **Cosine Similarity**.

### How to run it on the local machine?
1. Clone the repository using the command `git clone https://github.com/adityapradhan202/Grimoire-Guide.git`
2. Open a new terminal in the project folder and type this command:  
`pip install -r requirements.txt`
3. You will need the spacy large english model to run the recommendation system. Here's a quick tutorial for downloading the model. Open a terminal a write this command.
```
python -m spacy download en_core_web_lg
```
4. Nodejs with npm should be installed in your computer.
5. Now open a new terminal and run `npm install`. If you are encountering an error even after installing nodejs and npm properly, the problem might be the execution policy of the windows powershell. To bypass and run npm install command, type this the windows powershell:  
```
powershell -ExecutionPolicy Bypass -Command "npm install"
```  
This will install all the packages required for the web application.  
6. Now open two separate terminals, run the scripts [app.py](https://github.com/adityapradhan202/Binge-Trend/blob/main/Flask_api/app.py) and [app2.py](https://github.com/adityapradhan202/Binge-Trend/blob/main/Flask_api/app2.py) seperately to run the Flask APIs.  
7. To run the web application run `node index.js` in a terminal.  
8. Now the web is ready to use! Enjoy searching the movies and webseries of your type.  

### Data collection:
Data has been collected from [Rotten tomatoes](https://www.rottentomatoes.com/) and [IMDB](https://www.imdb.com/). Synopsis and plot of over 600 movies and webseries have been collected.

### Further improvements:
Neural networks can be used on big datasets to get better classification model. Pretrained models can be used to leverage the power of transfer learning, to get better results in less amount of data.





