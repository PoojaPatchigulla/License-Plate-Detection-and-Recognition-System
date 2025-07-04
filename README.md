License Plate Detection and Recognition System  Web App
This is a web-based application for automatic "License Plate Detection and Recognition (LPR)" using a deep learning model and a Flask backend.

🔧 Features
- Upload an image of a vehicle.
- Detect and recognize the license plate number.
- View results on a clean and simple web interface.
- Built using Flask, HTML/CSS, and a pre-trained PyTorch model.

📁 Project Structure:
- License_Plate/
  - app.py → Main Flask application
  - testing.py → Testing script
  - best.pt → Trained PyTorch model
  - LPRNet_custom1/ → Model definitions and utilities
  - static/
    - css/ → Styling
    - uploads/ → Uploaded images
  - templates/
    - index.html → Upload form
    - result.html → Results page

🚀 How to Run
Clone this repository:
   ```bash
git clone https://github.com/yourusername/License_Plate.git
cd License_Plate

