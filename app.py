from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify, send_from_directory
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import os
import sqlite3
from datetime import datetime
import uuid
from functools import wraps
from detector import AggressiveLicensePlateDetector, SimpleLicensePlateDetector
from ocr import LicensePlateOCR
import cv2
import numpy as np

app = Flask(__name__)
app.secret_key = 'your-secret-key-here-change-in-production'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['DATABASE'] = 'alpr.db'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Database initialization
def init_db():
    conn = sqlite3.connect(app.config['DATABASE'])
    cursor = conn.cursor()
    
    # Create users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create detections table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            image_filename TEXT NOT NULL,
            detected_plate TEXT,
            confidence_score REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    conn.commit()
    conn.close()

# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Routes
@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        conn = sqlite3.connect(app.config['DATABASE'])
        cursor = conn.cursor()
        cursor.execute('SELECT id, password_hash FROM users WHERE username = ?', (username,))
        user = cursor.fetchone()
        conn.close()
        
        if user and check_password_hash(user[1], password):
            session['user_id'] = user[0]
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password', 'error')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        
        if password != confirm_password:
            flash('Passwords do not match', 'error')
            return render_template('register.html')
        
        try:
            conn = sqlite3.connect(app.config['DATABASE'])
            cursor = conn.cursor()
            
            # Check if user already exists
            cursor.execute('SELECT id FROM users WHERE username = ? OR email = ?', (username, email))
            if cursor.fetchone():
                flash('Username or email already exists', 'error')
                return render_template('register.html')
            
            # Create new user
            password_hash = generate_password_hash(password)
            cursor.execute('INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)',
                         (username, email, password_hash))
            conn.commit()
            conn.close()
            
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
            
        except sqlite3.Error as e:
            flash('Registration failed. Please try again.', 'error')
            return render_template('register.html')
    
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out', 'info')
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html')

@app.route('/upload')
@login_required
def upload():
    return render_template('upload.html')

@app.route('/history')
@login_required
def history():
    return render_template('history.html')

# Initialize ALPR components
try:
    detector = AggressiveLicensePlateDetector()
    ocr = LicensePlateOCR()
    print("Aggressive Text-First ALPR components loaded successfully")
except Exception as e:
    print(f"Error loading ALPR components: {e}")
    detector = SimpleLicensePlateDetector()
    ocr = None

@app.route('/api/upload', methods=['POST'])
@login_required
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file:
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        try:
            # Use new robust detection pipeline
            detection_result = detector.detect_license_plate_with_ocr(filepath, ocr)
            
            detected_plate = detection_result.get('plate_number', 'NOT_DETECTED')
            confidence = detection_result.get('confidence', 0.0)
            annotated_image_path = detection_result.get('annotated_image_path', None)
            
            # Generate image URL for frontend
            image_url = None
            if annotated_image_path:
                image_url = f"/uploads/{os.path.basename(annotated_image_path)}"
            else:
                # Fallback to original image
                image_url = f"/uploads/{unique_filename}"
            
        except Exception as e:
            print(f"Error in ALPR processing: {e}")
            detected_plate = "PROCESSING_ERROR"
            confidence = 0.0
            image_url = f"/uploads/{unique_filename}"
        
        # Save to database
        conn = sqlite3.connect(app.config['DATABASE'])
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO detections (user_id, image_filename, detected_plate, confidence_score)
            VALUES (?, ?, ?, ?)
        ''', (session['user_id'], unique_filename, detected_plate, confidence))
        conn.commit()
        conn.close()
        
        return jsonify({
            'success': True,
            'detected_plate': detected_plate,
            'confidence': confidence,
            'image_url': image_url
        })

@app.route('/api/history')
@login_required
def get_history():
    conn = sqlite3.connect(app.config['DATABASE'])
    cursor = conn.cursor()
    cursor.execute('''
        SELECT id, image_filename, detected_plate, confidence_score, created_at
        FROM detections
        WHERE user_id = ?
        ORDER BY created_at DESC
    ''', (session['user_id'],))
    detections = cursor.fetchall()
    conn.close()
    
    history_list = []
    for detection in detections:
        history_list.append({
            'id': detection[0],
            'image_filename': detection[1],
            'detected_plate': detection[2],
            'confidence': detection[3],
            'created_at': detection[4],
            'image_url': f"/uploads/{detection[1]}"
        })
    
    return jsonify(history_list)

@app.route('/api/detection/<int:detection_id>', methods=['DELETE'])
@login_required
def delete_detection(detection_id):
    conn = sqlite3.connect(app.config['DATABASE'])
    cursor = conn.cursor()
    
    # Get detection info to delete the image file
    cursor.execute('SELECT image_filename FROM detections WHERE id = ? AND user_id = ?', 
                   (detection_id, session['user_id']))
    detection = cursor.fetchone()
    
    if detection:
        # Delete image file
        try:
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], detection[0])
            if os.path.exists(image_path):
                os.remove(image_path)
        except Exception as e:
            print(f"Error deleting image file: {e}")
        
        # Delete database record
        cursor.execute('DELETE FROM detections WHERE id = ? AND user_id = ?', 
                      (detection_id, session['user_id']))
        conn.commit()
        conn.close()
        
        return jsonify({'success': True})
    else:
        conn.close()
        return jsonify({'error': 'Detection not found'}), 404

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    init_db()
    app.run(debug=True, host='0.0.0.0', port=5000)
