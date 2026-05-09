import os
import json
from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash

from config import Config
from models import db, User, PredictionHistory
from forms import LoginForm, RegisterForm
from utils import allowed_file, save_uploaded_file
from predict import predict_plant
from gradcam import generate_gradcam

app = Flask(__name__)
Config.init_app(app)
app.config.from_object(Config)

# Initialize Database and Login Manager
db.init_app(app)
login_manager = LoginManager()
login_manager.login_view = 'login'
login_manager.init_app(app)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Create database tables
with app.app_context():
    db.create_all()

# Load medicinal uses JSON
try:
    with open('medicinal_uses.json', 'r') as f:
        medicinal_data = json.load(f)
except Exception:
    medicinal_data = {}

# ROUTES
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    form = RegisterForm()
    if form.validate_on_submit():
        hashed_password = generate_password_hash(form.password.data, method='pbkdf2:sha256')
        new_user = User(username=form.username.data, email=form.email.data, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        flash('Account created successfully! You can now log in.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html', form=form)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user and check_password_hash(user.password, form.password.data):
            login_user(user)
            flash('Logged in successfully.', 'success')
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for('dashboard'))
        else:
            flash('Login Unsuccessful. Please check email and password', 'danger')
    return render_template('login.html', form=form)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    # Fetch user prediction history
    history = PredictionHistory.query.filter_by(user_id=current_user.id).order_by(PredictionHistory.created_at.desc()).limit(10).all()
    return render_template('dashboard.html', history=history)

@app.route('/upload', methods=['POST'])
@login_required
def upload():
    if 'file' not in request.files:
        flash('No file part', 'danger')
        return redirect(url_for('dashboard'))
    file = request.files['file']
    if file.filename == '':
        flash('No selected file', 'danger')
        return redirect(url_for('dashboard'))
    if file and allowed_file(file.filename, app.config['ALLOWED_EXTENSIONS']):
        filename = save_uploaded_file(file, app.config['UPLOAD_FOLDER'])
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # 1. Predict
        prediction = predict_plant(file_path)
        predicted_class = prediction['predicted_class']
        confidence = prediction['confidence']
        
        # 2. Generate Grad-CAM
        gradcam_filename = generate_gradcam(file_path, filename)
        
        # 3. Save to database
        history_entry = PredictionHistory(
            user_id=current_user.id,
            image_filename=filename,
            predicted_class=predicted_class,
            confidence=confidence,
            gradcam_filename=gradcam_filename
        )
        db.session.add(history_entry)
        db.session.commit()
        
        return redirect(url_for('result', history_id=history_entry.id))
    
    flash('Allowed file types are png, jpg, jpeg', 'warning')
    return redirect(url_for('dashboard'))

@app.route('/result/<int:history_id>')
@login_required
def result(history_id):
    history = PredictionHistory.query.get_or_404(history_id)
    # Ensure user owns this history
    if history.user_id != current_user.id:
        flash("You do not have permission to view this result.", "danger")
        return redirect(url_for('dashboard'))
        
    plant_info = medicinal_data.get(history.predicted_class, None)
    return render_template('result.html', history=history, plant_info=plant_info)

@app.route('/library')
def library():
    return render_template('library.html', plants=medicinal_data)

@app.route('/science')
def science():
    return render_template('science.html')

@app.route('/portfolio')
def portfolio():
    return render_template('portfolio.html')

@app.route('/plant/<plant_name>')
def plant_detail(plant_name):
    plant_info = medicinal_data.get(plant_name)
    if not plant_info:
        flash('Plant not found in database.', 'danger')
        return redirect(url_for('library'))
    return render_template('plant_detail.html', plant_name=plant_name, info=plant_info)

if __name__ == '__main__':
    app.run(debug=True)
