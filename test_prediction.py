from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load trained model
model = load_model("dermacare_model.h5")

# Test image path
img_path = "Dataset/test/eczema/sample1.jpeg"  # apni image path yahan

# Load and preprocess image
img = image.load_img(img_path, target_size=(224,224))
img_array = np.expand_dims(np.array(img)/255.0, axis=0)

# Predict
pred = model.predict(img_array)
classes = ['acne', 'eczema', 'psoriasis', 'drug_rash']

print("Prediction:", classes[np.argmax(pred)])
print("Confidence:", pred[0][np.argmax(pred)]*100, "%")