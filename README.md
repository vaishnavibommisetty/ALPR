# 🚗 Automatic License Plate Recognition (ALPR) System

A full-stack web application for automatic license plate detection and recognition using YOLOv8 and Tesseract OCR.

## ✨ Features

- 🔐 **User Authentication**: Secure login, registration, and session management
- 📊 **Professional Dashboard**: Real-time statistics and system status
- 📸 **Image Upload**: Drag-and-drop interface for vehicle images
- 🤖 **AI-Powered Detection**: YOLOv8 for license plate localization
- 🔤 **OCR Text Extraction**: Tesseract OCR for plate number recognition
- 📈 **Detection History**: Complete history with filtering and export options
- 📱 **Responsive Design**: Mobile-friendly modern UI
- 💾 **Data Storage**: SQLite database for user data and detection results

## 🛠️ Technology Stack

### Backend
- **Flask** - Web framework
- **SQLite** - Database
- **OpenCV** - Image processing
- **YOLOv8** - Object detection
- **Tesseract** - OCR engine
- **Werkzeug** - Security utilities

### Frontend
- **Tailwind CSS** - Styling framework
- **Font Awesome** - Icons
- **Vanilla JavaScript** - Interactivity

## 📋 Prerequisites

- Python 3.8 or higher
- Tesseract OCR engine
- Git (for cloning)

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd ALPR
```

### 2. Install Tesseract OCR

**Windows:**
```bash
# Download and install from: https://github.com/UB-Mannheim/tesseract/wiki
# Make sure to add Tesseract to your system PATH
```

**macOS:**
```bash
brew install tesseract
```

**Linux:**
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr
```

### 3. Create Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 4. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 5. Run the Application
```bash
python app.py
```

The application will be available at `http://localhost:5000`

## 📁 Project Structure

```
ALPR/
├── app.py              # Main Flask application
├── models.py           # Database models and utilities
├── detector.py         # YOLOv8 license plate detection
├── ocr.py             # Tesseract OCR text extraction
├── requirements.txt    # Python dependencies
├── README.md          # This file
├── templates/         # HTML templates
│   ├── base.html
│   ├── login.html
│   ├── register.html
│   ├── dashboard.html
│   ├── upload.html
│   └── history.html
├── static/           # Static files (CSS, JS, images)
├── uploads/          # Uploaded images (auto-created)
└── alpr.db          # SQLite database (auto-created)
```

## 🎯 Usage

### 1. Register an Account
- Visit `http://localhost:5000/register`
- Create a new account with username, email, and password

### 2. Login
- Use your credentials to login at `http://localhost:5000/login`

### 3. Upload and Detect
- Navigate to the Upload page
- Drag and drop or select a vehicle image
- Click "Detect License Plate" to process the image
- View results with confidence scores

### 4. View History
- Access the History page to see all past detections
- Filter by plate number, date range, or confidence
- Export results to CSV

## 🔧 Configuration

### Environment Variables (Optional)
Create a `.env` file in the root directory:

```env
FLASK_SECRET_KEY=your-secret-key-here
FLASK_ENV=development
UPLOAD_FOLDER=uploads
DATABASE=alpr.db
MAX_CONTENT_LENGTH=16777216  # 16MB
```

### Customization Options

#### Detection Thresholds
In `detector.py`, modify confidence thresholds:
```python
if confidence > 0.5:  # Adjust this value
```

#### OCR Configuration
In `ocr.py`, customize Tesseract settings:
```python
self.custom_config = r'--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
```

## 🐛 Troubleshooting

### Common Issues

1. **Tesseract not found**
   - Ensure Tesseract is installed and in system PATH
   - Restart your terminal after installation

2. **YOLO model download issues**
   - The application will automatically download YOLOv8 models on first run
   - Ensure internet connection is available

3. **Database errors**
   - Delete `alpr.db` and restart the application
   - Ensure write permissions in the project directory

4. **Upload failures**
   - Check file size limit (16MB default)
   - Ensure supported image formats (JPG, PNG, BMP, GIF)

### Performance Tips

- Use GPU acceleration for YOLOv8 (install PyTorch with CUDA)
- Optimize image sizes before upload
- Consider using Redis for session storage in production

## 📊 API Endpoints

### Authentication
- `POST /login` - User login
- `POST /register` - User registration
- `GET /logout` - User logout

### Detection
- `POST /api/upload` - Upload and process image
- `GET /api/history` - Get user detection history
- `DELETE /api/detection/<id>` - Delete detection record

### Static Files
- `GET /uploads/<filename>` - Serve uploaded images

## 🔒 Security Features

- Password hashing with Werkzeug
- Session-based authentication
- File upload validation
- SQL injection prevention
- XSS protection with Flask
- CSRF protection (add Flask-WTF for production)

## 🚀 Deployment

### Production Setup

1. **Use a production WSGI server:**
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

2. **Environment variables:**
```bash
export FLASK_ENV=production
export FLASK_SECRET_KEY=your-secure-secret-key
```

3. **Database:**
- Consider PostgreSQL or MySQL for production
- Set up proper backups

4. **Web Server:**
- Use Nginx or Apache as reverse proxy
- Configure SSL certificates

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Ultralytics](https://ultralytics.com/) for YOLOv8
- [Tesseract](https://github.com/tesseract-ocr/tesseract) for OCR
- [Flask](https://flask.palletsprojects.com/) for the web framework
- [Tailwind CSS](https://tailwindcss.com/) for styling

## 📞 Support

For issues and questions:
- Create an issue on GitHub
- Check the troubleshooting section above
- Review the API documentation

---

**Built with ❤️ using Python, Flask, YOLOv8, and Tesseract OCR**
