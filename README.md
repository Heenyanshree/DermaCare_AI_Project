# DermaCare AI Project

## Description
**DermaCare AI** is a web application that predicts skin diseases from images.  
The app classifies images into **Acne, Drug Reaction, Eczema, or Psoriasis** and provides care suggestions to the user.

## Features
- Upload skin images and get predictions
- Confidence score for each prediction
- Care suggestions for each detected disease
- Dashboard showing analytics and disease frequency
- History of all predictions stored in database

## Tech Stack
- **Backend:** Python, Flask  
- **AI/ML:** TensorFlow, Keras  
- **Database:** SQLite  
- **Frontend:** HTML, CSS, 
- **Environment:** Python virtual environment (`derma_env`)  

## Installation & Setup
1. Clone the repository:  
   ```bash
   git clone https://github.com/Heenyanshree/DermaCare_AI_Project.git

2. Navigate into project directory:
          cd DermaCare_AI_Project

3. Create & activate virtual environment:

        python -m venv derma_env
       source derma_env/Scripts/activate   # For Windows

4. Install dependencies:

       pip install -r requirements.txt

5. Run the Flask app:

        python backend/app.py

6. Open your browser and go to:

        http://127.0.0.1:5000


Folder Structure
DermaCare_AI_Project/


│
├── backend/          # Flask backend and database
│   ├── app.py
│   ├── dermacare.db
│   └── templates/


│
├── frontend/         # Frontend files (HTML/CSS/JS)

│

├── model/            # Trained AI model
│   └── dermacare_model.h5


│
├── Dataset/          # Training dataset (optional)
├── derma_env/        # Virtual environment (ignored in Git)
├── static/           # Uploaded images and assets
├── train_model.py    # Model training script
├── test_prediction.py# Prediction testing script
└── requirements.txt

Usage

Upload a skin image in the app
View predicted disease with confidence score and suggestions
Check prediction history
Explore analytics dashboard





