from fastapi import FastAPI, File, UploadFile, HTTPException
from util import detect_id,segment_id
from PIL import Image
import pytesseract
import io


app = FastAPI()
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
@app.get("/")
def read_root():
    return {"message": "Welcome to the ID classification API"}

@app.get("/predict/")
async def predict_image():
    try:
       
        image = r"C:\Users\21626\Downloads\3_jpg.rf.a1872885a96c65c5ed2322111890c9aa.jpg"
       
        img_class = detect_id(image)
        if  img_class == "id":

            seg_img = segment_id(image)
            
               

            if not seg_img:
                raise HTTPException(status_code=400, detail="No regions detected in the image.")
            results = []
            for i, seg in enumerate(seg_img):
                 # Ensure that each segment is a PIL image
                if not isinstance(seg, Image.Image):
                    raise HTTPException(status_code=400, detail=f"Segment {i+1} is not a valid image object.")

                text = pytesseract.image_to_string(seg, lang='ara')
                results.append({"segment": i+1, "text": text})
                
                    
                    
                    
            # if not text:
            #     raise HTTPException(status_code=400, detail="No text detected in the image.")
            return { "id informations": results}
        
        else:
            
            return {"detail": "This image is not an ID"}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")
