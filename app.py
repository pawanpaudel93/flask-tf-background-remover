import os
import sys
from mrcnn import utils
from mrcnn import model as modellib
from flask import Flask, flash, request, render_template, redirect, session, url_for, send_from_directory, jsonify
from flask_dropzone import Dropzone
from flask_uploads import UploadSet, configure_uploads, IMAGES, patch_request_class
from werkzeug.utils import secure_filename
from decouple import config
import tensorflow as tf

import coco
from helpers import remove_bg
from model import get_model

app = Flask(__name__, static_url_path='/static')
dropzone = Dropzone(app)

app.secret_key = config("SECRET_KEY", cast=str, default="mynameispawan")
app.config['SESSION_TYPE'] = 'filesystem'

# Dropzone settings
app.config['DROPZONE_UPLOAD_MULTIPLE'] = True
app.config['DROPZONE_ALLOWED_FILE_CUSTOM'] = True
app.config['DROPZONE_ALLOWED_FILE_TYPE'] = 'image/*'
app.config['DROPZONE_REDIRECT_VIEW'] = 'results'

# Uploads settings
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOADED_IMAGES_DEST'] = os.getcwd() + '/static/images/input'
app.config['OUTPUT_IMAGES_DEST'] = os.getcwd() + '/static/images/output'
images = UploadSet('images', IMAGES)
configure_uploads(app, images)
patch_request_class(app)  # set maximum file size, default is 16MB

# model
graph = None
model = None

DEFAULT_CONFIG = {"BLACKnWHITE": False, "BG_WHITE": False}

@app.route('/initilize', methods=['GET'])
def initialize_model():
    global graph, model
    graph = tf.get_default_graph()
    model = get_model()
    return jsonify(success=True)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/images/<filename>')
def output_file(filename):
    return send_from_directory(app.config['OUTPUT_IMAGES_DEST'] + '/', filename)


@app.route('/switch/config', methods=['POST'])
def switch_config():
    if request.method == 'POST':
        data = request.get_json()
        if "bnw" in data:
            DEFAULT_CONFIG["BLACKnWHITE"] = bool(data["bnw"])
        if "bgwhite" in data:
            DEFAULT_CONFIG["BG_WHITE"] = bool(data["bgwhite"])
        return jsonify(success=True)


@app.route('/', methods=['GET', 'POST'])
def bg_remove():
    file_urls = []
    if request.method == 'POST':
        files_obj = request.files
        for f in files_obj:
            file = request.files.get(f)
            filename = images.save(
                file,
                name=file.filename
            )
            file_path = app.config['UPLOADED_IMAGES_DEST'] + '/' + filename
            output_filepath = app.config['OUTPUT_IMAGES_DEST'] + '/' + filename
            output_filename = remove_bg(model, file_path, output_filepath, graph, DEFAULT_CONFIG)
            # (images.url(filename))
            file_urls.append(url_for('output_file', filename=output_filename))
            session['file_urls'] = file_urls
            return "uploading..."
    if request.method == 'GET':
        return render_template('index.html')


@app.route('/results')
def results():
    # redirect to home if no images to display
    if "file_urls" not in session or session['file_urls'] == []:
        return redirect(url_for('bg_remove'))

    # set the file_urls and remove the session variable
    file_urls = session['file_urls']
    session.pop('file_urls', None)

    return render_template('results.html', file_urls=file_urls)


if __name__ == "__main__":
    app.secret_key = 'super secret key'
    app.config['SESSION_TYPE'] = 'filesystem'
    # sess.init_app(app)
    app.run(debug=True, port=os.getenv('PORT', 5000))
