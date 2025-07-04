

import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, url_for
from ultralytics import YOLO


from LPRNet_custom1.lpr_infer import recognize_plate

app = Flask(__name__)

# Load YOLO model
yolo_model = YOLO("best.pt")

def process_license_plate(image_path):
    """Use LPRNet to recognize text from the license plate image."""
    try:
        result = recognize_plate(image_path)
        return result if result else "No text detected"
    except Exception as e:
        return f"OCR Error: {str(e)}"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/detect", methods=["POST"])
def detect():
    if "file" not in request.files:
        return redirect(request.url)

    file = request.files["file"]
    if file.filename == "":
        return redirect(request.url)

    if file:
        filepath = "static/uploads/input.jpg"
        file.save(filepath)

        results = yolo_model(filepath)
        for result in results:
            for box in result.boxes.xyxy:
                x1, y1, x2, y2 = map(int, box.tolist())
                image = cv2.imread(filepath)
                cropped_plate = image[y1:y2, x1:x2]

                cropped_path = "static/uploads/cropped_plate.jpg"
                cv2.imwrite(cropped_path, cropped_plate)

                plate_text = process_license_plate(cropped_path)

                return render_template("result.html", input_image=filepath, plate_image=cropped_path, plate_text=plate_text)

    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True)
