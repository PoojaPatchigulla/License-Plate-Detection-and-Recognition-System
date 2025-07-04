# from ultralytics import YOLO
# import cv2

# # Load trained model
# model = YOLO("best.pt")

# # Run inference on an image
# image_path = r"A:\BTECH\6thsem\mini project\License 2\unwanted\India2\Datacluster_number_plates (18).jpg"  # Replace with an actual test image
# image = cv2.imread(image_path)

# results = model(image)

# # Show detected results
# for result in results:
#     result.show()


import cv2
import easyocr

reader = easyocr.Reader(['en'])

# Load manually cropped image
image_path = "static/uploads/cropped_plate.jpg"
image = cv2.imread(image_path)

if image is None:
    print("Error: Cropped image not found or not readable!")
else:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    result = reader.readtext(gray, detail=1)  # Get confidence scores

    if result:
        for bbox, text, conf in result:
            print(f"Detected: {text} (Confidence: {conf:.2f})")
    else:
        print("No text detected!")
