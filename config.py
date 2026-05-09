import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'a_very_secret_key_for_development_only'
    
    # SQLite Database Config
    basedir = os.path.abspath(os.path.dirname(__file__))
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or \
        'sqlite:///' + os.path.join(basedir, 'instance', 'users.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Upload Configurations
    UPLOAD_FOLDER = os.path.join(basedir, 'static', 'uploads')
    GRADCAM_FOLDER = os.path.join(basedir, 'static', 'gradcam')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB max upload
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    
    # ML Model Config
    MODEL_PATH = os.path.join(basedir, 'saved_model', 'medical_plant_model.keras')
    
    @staticmethod
    def init_app(app):
        # Create directories if they don't exist
        os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(Config.GRADCAM_FOLDER, exist_ok=True)
        os.makedirs(os.path.join(Config.basedir, 'saved_model'), exist_ok=True)
        os.makedirs(os.path.join(Config.basedir, 'instance'), exist_ok=True)
