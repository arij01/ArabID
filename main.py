from fastapi import FastAPI, File, UploadFile, HTTPException
from util import detect_id, segment_id
from PIL import Image
import pytesseract


app = FastAPI()
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

@app.get("/")
def read_root():
    return {"message": "Welcome to the ID classification API"}

@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    try:
        # Save the uploaded file to a temporary location
        image_data = await file.read()
        temp_image_path = "temp_image.jpg"  # You can use a different path or method

        with open(temp_image_path, "wb") as f:
            f.write(image_data)

        # Now use the path for processing
        img_class = detect_id(temp_image_path)
        if img_class == "id":
            seg_img = segment_id(temp_image_path)
            if not seg_img:
                raise HTTPException(status_code=400, detail="No regions detected in the image.")
            results = []
            for i, seg in enumerate(seg_img):
                # Ensure that each segment is a PIL image
                if not isinstance(seg, Image.Image):
                    raise HTTPException(status_code=400, detail=f"Segment {i+1} is not a valid image object.")
                text = pytesseract.image_to_string(seg, lang='ara')
                results.append({"segment": i+1, "text": text})
            return {"id informations": results}
        else:
            return {"detail": "This image is not an ID"}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")
