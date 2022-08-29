from ast import main
import os
import shutil
from flask import Flask, render_template, request, redirect, url_for,jsonify
import base64
from predict import main, process64



sourceImg = "static/source.jpg"
app = Flask(__name__)
@app.route('/', methods=['GET','POST'])
def index ():
    if(request.method == 'GET'):
        return render_template("upload.html")

    elif(request.method == 'POST'):
        image = request.files['image']
        image.save(sourceImg)
        data = main(sourceImg)
        # return render_template("upload.html")
        return render_template('result.html',data=data)

@app.route('/api/', methods=['POST'])
def upload():
    respond = []
    if request.method == 'POST':
        imgstring = request.json.get("image")
        image_in_string = base64.b64decode(imgstring)
        verdic_data = process64(image_in_string)
        respond.append(verdic_data)

        return jsonify(respond)


    else:
        respond.append({
            "message":"request error"
        })
    return jsonify(respond)


if __name__ == '__main__':
    os.makedirs('static', exist_ok=True)
    app.run(host='0.0.0.0', port=5000)
    shutil.rmtree('static')