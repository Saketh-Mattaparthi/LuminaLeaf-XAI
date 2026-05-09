# LuminaLeaf XAI: Explainable Medicinal Plant Identification

![UI Preview](https://via.placeholder.com/1200x600.png?text=LuminaLeaf+XAI+Hero+Section)

An advanced, premium full-stack AI/ML application that identifies the medicinal use of plants from their leaf images using a fine-tuned InceptionV3 model. Built with a Flask backend, SQLite database, and a beautiful "glassmorphic botanical" frontend.

## 🌟 Features
*   **Advanced AI Prediction**: Uses Transfer Learning (InceptionV3) to classify plant species with high accuracy.
*   **Explainable AI**: Generates Grad-CAM heatmaps so users can see *which parts* of the leaf the AI focused on.
*   **Medicinal Knowledge Base**: Provides scientific names, traditional uses, and toxicity warnings.
*   **Premium UI/UX**: Built with Bootstrap 5, AOS animations, and a modern glassmorphism design system.
*   **Authentication & Dashboard**: Secure user registration, login, and a personalized dashboard with prediction history.

## 🛠️ Tech Stack
*   **Frontend**: HTML5, CSS3 (Vanilla/Glassmorphism), JavaScript, Bootstrap 5, Chart.js.
*   **Backend**: Python, Flask, Flask-Login, Flask-SQLAlchemy, Flask-WTF.
*   **Machine Learning**: TensorFlow / Keras, OpenCV, NumPy, Matplotlib.
*   **Database**: SQLite (Perfect for interview demos and rapid deployment).

## 🚀 Installation & Setup

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/medical_plant_ai.git
    cd medical_plant_ai
    ```

2.  **Create a virtual environment & install dependencies**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    pip install -r requirements.txt
    ```

3.  **Run the application**:
    ```bash
    python app.py
    ```
    *The SQLite database and necessary folders will be initialized automatically.*

4.  **View the app**:
    Open `http://127.0.0.1:5000` in your browser.

## 🧠 Model Training Pipeline

To train the model yourself:
1.  Place your dataset in a `data/train` and `data/validation` structure, where subfolders represent class names.
2.  Update `labels.json` and `medicinal_uses.json` to match your classes.
3.  Run the training script:
    ```bash
    python train_model.py
    ```
    *Note: The script uses `tf.data` for efficient data loading, heavy augmentation, and callbacks like `EarlyStopping` and `ReduceLROnPlateau`.*

## 💡 Interview / Defense Q&A

**Q: Why Flask instead of Django or FastAPI?**
A: Flask is lightweight, highly customizable, and perfectly suited for serving machine learning models quickly. It allowed me to structure the application exactly how I wanted without the overhead of a heavy framework.

**Q: Why InceptionV3?**
A: InceptionV3 strikes a great balance between accuracy and computational efficiency. Its multi-scale spatial feature extraction (Inception modules) is particularly good at capturing the intricate vein patterns and shapes of leaves.

**Q: What is Grad-CAM?**
A: Gradient-weighted Class Activation Mapping (Grad-CAM) is an Explainable AI (XAI) technique. It highlights the regions of the image that were most important for the model's prediction, ensuring the model isn't just "guessing" based on background noise.

**Q: How is security handled?**
A: User passwords are mathematically hashed using Werkzeug security. Session management is handled securely via Flask-Login. Image uploads are strictly validated (only `.png`, `.jpg`, `.jpeg`) to prevent malicious file execution.
