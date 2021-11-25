# Inference uploaded job description using deployed NER model on web via flask
"""
# USAGE
# Start the server:
# 	python ner_app.py
# Submit a request via cURL:
# 	curl -X POST -F text=@ad.txt 'http://localhost:5000/'

""" 
# import the necessary packages
from flask import Flask, request, render_template
import spacy
import os


# initialize Flask application and the spacy model
app = Flask(__name__)


model_path = 'output/vector_based'
trained_nlp = spacy.load(os.path.join(model_path, "best_model"))
print('NER model loaded')

text = str()
loginName = str()

def prepare_text(filename):

    text = str()
    with open(filename, 'r') as f:
        contents = f.readlines()
        for content in contents:
            text = text + content

    return text

@app.route('/')
def to_login_page():
    return render_template("login.html")


@app.route("/login", methods=["POST"])
def login():
    global loginName
    
    loginName = request.form["loginName"]
    loginPwd = request.form["loginPwd"]
    if loginName == "admin" and loginPwd == "admin":
        print("login success")
        return render_template("upload.html", loginName=loginName)
    else:
        return render_template("login.html", msg="Wrong username or password. Try again")

# upload text file for prediction
@app.route('/upload_file', methods=['POST'])
def upload_file():
    global text
    file = request.files["file"]
    file_content = file.read()
    text = file_content.decode("utf-8")
    
    return render_template("predict.html", input_text=text)

# predict input text
@app.route('/predict_file', methods=['POST'])
def predict_file():
    
    # 'predict' button is pressed
    if 'predict' in request.form:
        global text
        
        doc = trained_nlp(text)
        prediction = []

        # extract all entities and its labels in the text
        if len(doc.ents) == 0:
            print ("No entities found.")
        else:
            for ent in doc.ents:
                prediction.append({'text':ent.text, 'entity':ent.label_})
                print (ent.text, ent.label_)
            
        results = str(prediction)
        return render_template("predict.html", input_text=text, results=results)
    
    # 'back' button is pressed
    elif 'back' in request.form:
        global loginName
        return render_template("upload.html", loginName=loginName)

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)