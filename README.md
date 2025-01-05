# Liver Guard

**Liver Guard** is a Flask-based web application designed for liver cancer prediction and analysis. It uses machine learning models for text-based predictions and deep learning models for CT scan segmentation to provide a comprehensive liver health diagnostic tool.

---

## Features

1. **Blood Test Input Prediction**:
   - Enter your blood test parameters to receive a prediction on liver cancer likelihood.
   - Uses an ensemble machine learning model for accurate predictions.

2. **CT Scan Image Analysis**:
   - Upload CT scan images for liver segmentation.
   - Provides masked outputs to identify liver and potential anomalies using a pre-trained model.

3. **User Management**:
   - Sign-up and log-in functionalities for personalized usage.
   - Secure password storage with hashing.

4. **Responsive Interface**:
   - Intuitive and user-friendly design for both blood test inputs and CT scan analysis.

---

## Technology Stack

- **Backend**: Flask, Python
- **Frontend**: HTML, CSS, JavaScript (with Flask's Jinja templating)
- **Machine Learning**: Ensemble model for liver cancer prediction
- **Deep Learning**: ResNet-50 model for liver segmentation
- **Database**: SQLite

---

## Project Structure

```
Liver Guard/
│
├── models/                  # Trained models
│   ├── predictions.pkl      # Blood Test Input Prediction
│   └── Cancer_Detection.pkl # CT Scan
│
├── templates/               # HTML templates for Flask
│   ├── index.html           # Homepage template
│   ├── home.html            # Dashboard template after login
│   ├── bloodtestinput.html  # Blood test prediction input page
│   ├── ctscan.html          # CT scan upload page
│   └── result.html          # Result display page
│
├── static/                  # Static files (CSS, JS, Images)
│   └── stles.css/           # JavaScript files (if any)
│
├── uploads/                 # Folder for storing user-uploaded images
│
├── results/                 # Folder for storing generated segmentation masks
│
├── app.py                   # Main Flask application script
├── detect.py                # CT scan processing and mask generation logic
├── predict.py               # Blood test prediction logic
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```

## Setup Instructions

### Prerequisites
- Python 3.x  
- Pip package manager  
- Git  

---

### Installation

#### 1. Clone the Repository
```bash
git clone https://github.com/your_username/LiverGuard.git
cd LiverGuard
```
#### 2. Set Up a Virtual Environment
```bash
python -m venv venv
# Activate the virtual environment
source venv/bin/activate   # On Windows, use `venv\Scripts\activate`
```

#### 3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Prepare the Application
1. Make sure the `models/` folder is present.
2. Download the following model files from the provided drive links:
   - Cancer_Detection.pkl and predictions.pkl from [Link](https://drive.google.com/drive/folders/1y6vjfEYjr5lOQkoBkqEET5bAoI8hDLZR?usp=sharing)
   
   Place them into the `models/` folder.

---

## Run the Application
```bash
python app.py
```

---

## Access the Application
- Open your browser and go to [http://127.0.0.1:5000](http://127.0.0.1:5000).

---

## Usage

### 1. Account Management
1. Use the **Sign Up** page to create a new account.
2. Log in to access personalized functionalities.
   
### 2. Blood Test Prediction
1. Navigate to the **Blood Test Input** page.
2. Enter all required parameters.
3. Click **Submit Prediction** to get the prediction.
4. The results will be displayed on the same page.

### 3. CT Scan Analysis
1. Go to the **CT Scan Upload** page.
2. Click **Browse Your Files**
3. Upload a valid CT scan image.
4. Click **Upload** to process the image.
5. Redirects automatically to the Results page where the original scan and its corresponding segmentation mask are visible.

---

## Screenshots

### Login Page
![Screenshot 2025-01-05 122944](https://github.com/user-attachments/assets/cca7c6a8-5413-47e9-90e4-4f485df789d6)

### Main Page
![Screenshot 2025-01-05 123033](https://github.com/user-attachments/assets/8cf83745-854a-42e6-807b-9988ac15daca)

### Blood Test Page
![Screenshot 2025-01-05 123047](https://github.com/user-attachments/assets/83e97f59-2056-4b85-90d6-ac38cb40ce1f)

### Blood Test Output
![Screenshot 2025-01-05 123111](https://github.com/user-attachments/assets/36ee652e-3734-4b39-85a0-08633d0233d0)

### CT Scan Page
![Screenshot 2025-01-05 123125](https://github.com/user-attachments/assets/db2ad875-828e-4f58-9b3d-f12df14f847b)

### Upload Preview
![Screenshot 2025-01-05 123147](https://github.com/user-attachments/assets/27ecc57a-90a5-42d4-aed3-e8103cf31821)

### CT Scan Results Page
![Screenshot 2025-01-05 123200](https://github.com/user-attachments/assets/b3b3e410-9238-4479-8f41-f1d05b2d2e6e)
