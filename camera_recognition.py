import cv2
import torch
from torchvision import transforms
from model import SimpleCNN

# Load the model
model = SimpleCNN()
model.load_state_dict(torch.load('/Users/tung/Desktop/model_weights.pth'))
model.eval()

# Preprocessing function
def preprocess_image(image):
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return preprocess(image).unsqueeze(0)

# Load Haar cascades classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open camera feed
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect faces
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Crop and preprocess face for model prediction
        face = frame[y:y+h, x:x+w]
        input_tensor = preprocess_image(face)
        
        with torch.no_grad():
            output = model(input_tensor)
            prediction = torch.argmax(output, dim=1)

        # Map prediction to text
        gender = "Male" if prediction.item() == 1 else "Female"
        
        # Draw bounding box and display prediction
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, gender, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('Gender Recognition', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
