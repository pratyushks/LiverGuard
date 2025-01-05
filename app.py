from flask import Flask, render_template, request, redirect, url_for, session, flash, send_from_directory
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import sqlite3
import pandas as pd
import os
from predict import predict_from_csv
from fastai.vision.all import *
from pathlib import Path
import pathlib
from detect import generate_mask


app = Flask(__name__)
app.secret_key = 'secretkey'

USER_INPUT_CSV = "user_inputs.csv"
MODEL_FILE1 = "models/predictions.pkl"
PREDICTION_OUTPUT_CSV = "predictions_output.csv"

UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
ALLOWED_EXTENSIONS = {'jpg','jpeg', 'png'}  

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER

MODEL_FILE = os.path.join(os.path.dirname(__file__), "Models", "Cancer_Detection.pkl")

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

def init_db():
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT NOT NULL,
            password TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()
    
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_x(fname: Path): 
    return fname

def label_func(x): 
    return 'train_masks' / f'{x.stem}_mask.png'

def foreground_acc(inp, targ, bkg_idx=0, axis=1):
    "Computes non-background accuracy for multiclass segmentation"
    targ = targ.squeeze(1)
    mask = targ != bkg_idx
    return (inp.argmax(dim=axis)[mask] == targ[mask]).float().mean()

def cust_foreground_acc(inp, targ):
    "Includes background in the accuracy computation"
    return foreground_acc(inp=inp, targ=targ, bkg_idx=3, axis=1) 

learn = load_learner(MODEL_FILE)
print("Model loaded successfully!")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/signup', methods=['POST'])
def signup():
    username = request.form['username']
    email = request.form['email']
    password = request.form['password']
    
    hashed_password = generate_password_hash(password)
    
    try:
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        cursor.execute('INSERT INTO users (username, email, password) VALUES (?, ?, ?)', (username, email, hashed_password))
        conn.commit()
        conn.close()
        flash('Sign-up successful! Please log in.', 'success')
    except sqlite3.IntegrityError:
        flash('Username already exists. Please log in.', 'error')
    return redirect(url_for('index'))

@app.route('/signin', methods=['POST'])
def signin():
    username = request.form['username']
    password = request.form['password']
    
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('SELECT password FROM users WHERE username = ?', (username,))
    result = cursor.fetchone()
    conn.close()
    
    if result and check_password_hash(result[0], password):
        session['user'] = username
        flash('Welcome back!', 'success')
        return redirect(url_for('home'))
    else:
        flash('Invalid username or password.', 'error')
        return redirect(url_for('index'))

@app.route('/home')
def home():
    if 'user' not in session:
        flash('Please log in to access the home page.', 'error')
        return redirect(url_for('index'))

    return render_template('home.html', username=session['user'])

@app.route('/bloodtestinput', methods=['GET', 'POST'])
def bloodtestinput():
    if request.method == 'POST':
        input_data = {
            'Age': request.form.get('age', type=int),
            'Gender': request.form.get('gender', type=int),
            'Total Bilirubin': request.form.get('total_bilirubin', type=float),
            'Direct Bilirubin': request.form.get('direct_bilirubin', type=float),
            'Alkphos Alkaline Phosphotase': request.form.get('alkphos', type=int),
            'Sgpt Alamine Aminotransferase': request.form.get('sgpt', type=int),
            'Sgot Aspartate Aminotransferase': request.form.get('sgot', type=int),
            'Total Proteins': request.form.get('total_proteins', type=float),
            'ALB Albumin': request.form.get('albumin', type=float),
            'A/G Ratio': request.form.get('ag_ratio', type=float),
        }

        user_inputs_df = pd.DataFrame([input_data])
        if os.path.exists(USER_INPUT_CSV):
            user_inputs_df.to_csv(USER_INPUT_CSV, mode='a', header=False, index=False)
        else:
            user_inputs_df.to_csv(USER_INPUT_CSV, index=False)

        try:
            user_inputs_df = pd.read_csv(USER_INPUT_CSV)
            predictions_df = predict_from_csv(user_inputs_df, MODEL_FILE1)

            predictions_df.to_csv(PREDICTION_OUTPUT_CSV, index=False)

            last_prediction = predictions_df.iloc[-1]['Prediction']
            return render_template('bloodtestinput.html', result=last_prediction)
        except Exception as e:
            flash(f"An error occurred: {e}", 'error')

        return redirect('/bloodtestinput')

    return render_template('bloodtestinput.html', result="")


@app.route('/ctscan', methods=['GET', 'POST'])
def ctscan():
    """Handle image upload and mask generation."""
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part', 'error')
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            flash('No selected file', 'error')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            # Save the uploaded file
            filename = secure_filename(file.filename)
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(upload_path)

            # Generate mask and save it to the results folder
            mask_filename = f"mask_{filename}"
            mask_path = os.path.join(app.config['RESULTS_FOLDER'], mask_filename)
            try:
                generate_mask(upload_path, mask_path)
            except Exception as e:
                flash(f"Error generating mask: {e}", 'error')
                return redirect(request.url)

            # Redirect to result page and pass both image and mask paths
            return redirect(url_for('result', filename=filename, mask_filename=mask_filename))

        flash('Unsupported file format. Please upload a JPG or PNG file.', 'error')
        return redirect(request.url)

    return render_template('ctscan.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve the uploaded file."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/results/<filename>')
def result_file(filename):
    """Serve the generated mask file."""
    return send_from_directory(app.config['RESULTS_FOLDER'], filename)

@app.route('/result')
def result():
    """Render the result page with the uploaded image and generated mask."""
    filename = request.args.get('filename')  # Get the uploaded image filename
    mask_filename = request.args.get('mask_filename')  # Get the generated mask filename

    if not filename or not mask_filename:
        flash('Missing files to display', 'error')
        return redirect(url_for('ctscan'))

    # Construct URLs for the uploaded image and generated mask
    image_path = url_for('uploaded_file', filename=filename)
    mask_path = url_for('result_file', filename=mask_filename)
    return render_template('result.html', original_image=image_path, mask_image=mask_path)


@app.route('/logout')
def logout():
    session.pop('user', None)
    flash('You have been logged out.', 'success')
    return redirect(url_for('index'))

if __name__ == '__main__':
    init_db()
    app.run(debug=True)
    
pathlib.PosixPath = temp
