from flask import Flask , request, url_for, render_template
import pickle
import numpy as np

def word_to_char(input):
    character=[]
    for i in input:
        character.append(i)
    return character

model = pickle.load(open('model.pkl' , 'rb'))
vector = pickle.load(open('vector.pkl' , 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')



@app.route('/check' , methods=['GET' , 'POST'])
def thank_you():
    enteredPassword = request.args.get('password')
    x_predict = np.array([enteredPassword])
    x_predict = vector.transform(x_predict)
    y_pred = model.predict(x_predict)

    return render_template('main.html', y_pred=y_pred)
    
    
if __name__ == '__main__':
    app.run(debug=True)