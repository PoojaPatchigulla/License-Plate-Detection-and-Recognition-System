License Plate Detection and Recognition Web App
This is a web-based application for automatic License Plate Recognition (LPR) using a deep learning model and a Flask backend.

🔧 Features
Upload an image of a vehicle.
Detect and recognize the license plate number.
View results on a clean and simple web interface.
Built using Flask, HTML/CSS, and a pre-trained PyTorch model.
🗂 Project Structure
License_Plate/
├── app.py                # Main Flask application
├── testing.py            # Testing script
├── best.pt               # Trained PyTorch model for LPR
├── LPRNet_custom1/       # Contains model definition and utilities
├── static/
│   ├── css/              # CSS styles
│   └── uploads/          # Uploaded images
├── templates/
│   ├── index.html        # Upload form page
│   └── result.html       # Results display page

🚀 How to Run
Clone this repository:
git clone https://github.com/yourusername/License_Plate.git
cd License_Plate
Install dependencies:
pip install -r requirements.txt
Run the Flask app:
python app.py
Open http://127.0.0.1:5000 in your browser.
🧠 Model
best.pt: Pre-trained LPRNet model used for license plate recognition.
Model logic is inside LPRNet_custom1/.


