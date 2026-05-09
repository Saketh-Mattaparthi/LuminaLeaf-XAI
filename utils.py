import os
from werkzeug.utils import secure_filename
import uuid

def allowed_file(filename, allowed_extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

def save_uploaded_file(file, upload_folder):
    """
    Saves file with a secure, unique UUID name to prevent collisions.
    Returns the secure filename.
    """
    original_filename = secure_filename(file.filename)
    extension = original_filename.rsplit('.', 1)[1].lower()
    
    unique_filename = f"{uuid.uuid4().hex}.{extension}"
    file_path = os.path.join(upload_folder, unique_filename)
    
    file.save(file_path)
    return unique_filename
