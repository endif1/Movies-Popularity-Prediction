import joblib
from flask import Flask, render_template, request
import pandas as pd
import json

app = Flask(__name__)

@app.route('/dataset')
def dataset():
    df = pd.read_csv(r'D:\DS Purwadhika\Final Project\rotten_tomato_dashboard.csv')
    return render_template('dataset.html',df_view = df.head(100))
@app.route('/analyze', methods=['POST'])
def analyze():
    if request.method == 'POST': 
        Content_rating  = request.form['content_rating']
        Runtime = request.form['runtime' ]
        Production = request.form['production_company_popularity']
        Actor = request.form['actor_popularity']
        Director = request.form['director_popularity']
        Author = request.form['author_popularity']
        Main_Genre = request.form['Main_Genre']
        Tomato_top = request.form['tomatometer_top_critics_count']
        Tomato_fresh = request.form['tomatometer_fresh_critics_count']
        Tomato_rotten = request.form['tomatometer_rotten_critics_count']

        Runtime = float(Runtime)
        Tomato_top = int(Tomato_top)
        Tomato_fresh = int(Tomato_fresh)
        Tomato_rotten = int(Tomato_rotten)

        data = {  
            'content_rating' : Content_rating,                      
            'runtime' : Runtime,
            'production_company_popularity' : Production,
            'actor_popularity' : Actor,
            'director_popularity' : Director,
            'author_popularity' : Author,
            'Main_Genre' : Main_Genre,
            'tomatometer_top_critics_count' : Tomato_top,
            'tomatometer_fresh_critics_count' : Tomato_fresh,
            'tomatometer_rotten_critics_count' : Tomato_rotten
        }

    # model = joblib.load(r'D:\belajar data scientist\Purwadhika\Flask Final Project\Model_DT_tuned_vers2')

    # df = pd.read_csv(r'D:\DS Purwadhika\Final Project\rotten_tomato_dashboard.csv')
    df_predict = pd.DataFrame(data = data, index = [1])
    model = joblib.load(r'D:\DS Purwadhika\Final Project\Model_XGB_tuned')
    prediction = model.predict(df_predict)
    predict_fix= round(prediction[0],2)
    print('Audience : ',predict_fix)
    return render_template('index.html',Content_rating=Content_rating,
                Runtime=Runtime,
                Production=Production,
                Actor=Actor,
                Director=Director,
                Author=Author,
                Main_Genre=Main_Genre,
                Tomato_top=Tomato_top,
                Tomato_fresh=Tomato_fresh,
                Tomato_rotten=Tomato_rotten,
                predict_fix = predict_fix) 

    
if __name__ == "_main_":

    app.run(debug=True)
