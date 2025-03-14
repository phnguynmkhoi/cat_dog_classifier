import torch
import io
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

import uvicorn
from torchvision import transforms

# Define the transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load model
try:
    model = torch.load("model/model_final.pth", weights_only=False)
    model.eval()
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
async def root():
    return {"message": "Welcome to the Dog-Cat Classification API!"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # try: 
    #     if not file.content_type.startswith('image/'):
    #         raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")
    # except Exception as e:
    #     # print(f"Error reading file type: {e}")
    #     raise HTTPException(status_code=400, detail=f"Error reading file type: {e}. File type: {file}")

    try:
        content = await file.read()
        image = Image.open(io.BytesIO(content)).convert("RGB")
        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

        with torch.no_grad():
            outputs = model(image_tensor)
            _, predicted = torch.max(outputs, 1)
            prediction_idx = predicted.item()
            
            # Map the prediction index to class label
            class_label = "dog" if prediction_idx == 0 else "cat"
            
            # Get probability
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence = probabilities[prediction_idx].item()
            
        return {
            "prediction": class_label,
            "confidence": round(confidence * 100, 2),
            "class_probs": {
                "dog": round(probabilities[0].item() * 100, 2),
                "cat": round(probabilities[1].item() * 100, 2)
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)