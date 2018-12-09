"""
Module to run the Flask app or just set it up for serving in Heroku
"""
import shutil
import sys
import os
import glob
import base64
import requests
from flask import Flask
from flask import render_template, request, redirect, flash, send_from_directory
from keras.preprocessing import image
sys.path.append(os.path.join(os.path.split(__file__)[0], '../models'))
import dog_detector
import face_detector 
import breed_classifier

UPLOAD_FOLDER = os.path.join(os.path.split(__file__)[0], 'uploads')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'aagiss@gmail.com#DSND'
FILES = list(sorted(glob.glob('uploads/*.jpg')))[-20:]
PREDICTIONS = []
for img_path in FILES:
    has_dog = 1 if dog_detector.dog_detector(img_path) else 0
    face_count = face_detector.face_detector(img_path)
    breed = breed_classifier.breed_classifier(img_path)
    prediction = {'has_dog': has_dog, 'face_count': face_count, 'breed': breed,
                  'google_query': 'https://www.google.gr/search?q=%s&tbm=isch' % breed.replace('_', '+')}
    PREDICTIONS.append(prediction)


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    """
    Index page serve
    """
    # render web page with plotly graphs
    return render_template('master.html',
                           recent_predictions=reversed(zip(FILES[-12:], 
                                                             PREDICTIONS[-12:])))


@app.route('/upload', methods=['POST'])
def upload():
    """
    The image processing endpoint
    """
    global FILES, PREDICTIONS
    is_file_upload = True
    if not 'query' in request.files:
        is_file_upload = False
    else:
        image_file = request.files['query']
        if image_file.filename == '':
            is_file_upload = False
    if not is_file_upload and not request.form.get('url'): 
        flash('ERROR: no file selected', 'error')
        return redirect('/index')
    tmp_filename = '%05d.tmp' % len(FILES)
    img_filename = '%05d.jpg' % len(FILES)
    tmp_path = os.path.join(app.config['UPLOAD_FOLDER'], tmp_filename)
    img_path = os.path.join(app.config['UPLOAD_FOLDER'], img_filename)
    if is_file_upload:
        image_file.save(tmp_path)
        img_url = os.path.join('uploads', img_filename)
    else:
        try:
            img_url = request.form.get('url')
            if img_url.startswith('data:image/jpeg;base64,'):
                coded_string = img_url[len('data:image/jpeg;base64,'):]
                raw_data = base64.b64decode(coded_string)
                with open(tmp_path, 'wb') as f:
                    f.write(raw_data)
                img_url = os.path.join('uploads', img_filename)
            else:
                r = requests.get(request.form.get('url'), stream=True)
                if r.status_code == 200:
                    with open(tmp_path, 'wb') as f:
                        r.raw.decode_content = True
                        shutil.copyfileobj(r.raw, f)
                else:
                    flash('ERROR: Could get url "%s": HTTP STATUS %d' % (request.form.get('url'), r.status_code), 'error')
                    return redirect('/index')
        except Exception as e:
           flash('ERROR: Could get url "%s": ' % request.form.get('url')+str(e), 'error')
           return redirect('/index')
    try:
        img = image.load_img(tmp_path, target_size=(224, 224))
    except Exception as e:
        os.unlink(tmp_path)
        flash('ERROR: Could not handle image: '+str(e), 'error')
        return redirect('/index')
    if img is None:
        os.unlink(tmp_path)
        flash('ERROR: Could not handle image', 'error')
        return redirect('/index')
    img.save(img_path)
    os.unlink(tmp_path)
    import dog_detector
    import face_detector
    import breed_classifier

    has_dog = 1 if dog_detector.dog_detector(img_path) else 0
    face_count = face_detector.face_detector(img_path)
    if not has_dog and face_count == 0:
        os.unlink(img_path)
        flash('ERROR: No dog or human face detected in image', 'error')
        return redirect('/index')
    breed = breed_classifier.breed_classifier(img_path)
    prediction = {'has_dog': has_dog, 'face_count': face_count, 'breed': breed, 'img_path': img_url,
                  'google_query': 'https://www.google.gr/search?q=%s&tbm=isch' % 
                                  breed.replace('_', '+')}
    FILES.append(img_url)
    PREDICTIONS.append(prediction)
    flash(prediction, 'prediction')
    return redirect('/index')

@app.route('/uploads/<path:path>')
def send_image(path):
    """
    Serve static images
    """
    return send_from_directory('uploads', path)

def main():
    """
    Main function to run standalone Flask App
    """
    app.debug = True
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
