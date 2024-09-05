from fastapi import FastAPI, File, UploadFile, HTTPException
from util import IDDetector, IDSegmenter
import pytesseract
import re
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize



app = FastAPI()

detector = IDDetector('classification.pt')
segmenter = IDSegmenter('segmentation.pt')
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def post_process_text(text):
    
    #cleaned_text = re.sub(r'[^أ-ي١-٩\s0-9a-zA-Z]', '', text)
    
    
    cleaned_text = re.sub(r'[\u0610-\u061A\u064B-\u065F]', '', text)
    
   
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    

    tokens = word_tokenize(cleaned_text)
  
    stop_words = set(stopwords.words('arabic') + stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    cleaned_text = ' '.join(tokens)
  
    return cleaned_text

@app.get("/")
def read_root():
    return {"message": "Welcome to the ID classification API"}

@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    try:
        
        image_data = await file.read()
        temp_image_path = "temp_image.jpg"  

        with open(temp_image_path, "wb") as f:
            f.write(image_data)

      

        
        img_class = detector.detect(temp_image_path)
        if img_class == "id":
            seg_img = segmenter.segment(temp_image_path)
            if not seg_img:
                raise HTTPException(status_code=400, detail="No regions detected in the image.")
            results = []
            custom_config = r'--oem 3 --psm 11'

            for i, seg in enumerate(seg_img):
                
                text = pytesseract.image_to_string(seg, lang='ara+eng+ara_number',config= custom_config )
                cleaned_text = post_process_text(text)
                results.append(cleaned_text)
            return {"id informations": results}
        else:
            return {"detail": "This image is not an ID"}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")