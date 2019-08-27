from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.feature_extraction.text import CountVectorizer



app = Flask(__name__)
m = pickle.load(open('mdl.pkl', 'rb'))
vtl = pickle.load(open('vocabulary_to_load.pkl', 'rb'))

cv = CountVectorizer(stop_words = 'english',vocabulary=vtl)

@app.route('/')
def home():
    return render_template('idx.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    tranform = request.form['text']
    #output = tranform
    tranform = tranform.split('\n')
    print(tranform)
    tranforming = cv.transform(tranform)
    pred_class_p = m.predict(tranforming)
    pred_prob_p = m.predict_proba(tranforming)
    if pred_prob_p[0,0] < pred_prob_p[0,1]:
        output = 'Spam Message'
    else:
        output = 'Not Spam Message'

    return render_template('idx.html', prediction_text='Sentence entered {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)
