from fastapi import FastAPI, File, UploadFile, HTTPException
from util import IDDetector, IDSegmenter
import pytesseract
import re
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import defaultdict
import cv2

app = FastAPI()
# Usage
detector = IDDetector('classification.pt')
segmenter = IDSegmenter('segmentation.pt')
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Define a dictionary to map common OCR errors to their corrections
# correction_dict = defaultdict(str)
# correction_dict['ا'] = 'أ'
# correction_dict['آ'] = 'أ'
# correction_dict['إ'] = 'إ'
# correction_dict['ؤ'] = 'ؤ'
# correction_dict['ئ'] = 'ئ'

# Define a function to preprocess the image
def preprocess_image(image_path):
    # You can add image preprocessing steps here, such as binarization, deskewing, etc.
    return image_path

# Define a function to postprocess the OCR output
def post_process_text(text):
    # Remove non-Arabic characters
    cleaned_text = re.sub(r'[^أ-ي\s]', '', text)
    
    # Remove diacritics
    cleaned_text = re.sub(r'[\u0610-\u061A\u064B-\u065F]', '', cleaned_text)
    
    # Remove extra spaces
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    
    # Correct common OCR errors
    # for key, value in correction_dict.items():
    #     cleaned_text = cleaned_text.replace(key, value)
    
    # Tokenize the text
    tokens = word_tokenize(cleaned_text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('arabic'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Join the tokens back into a string
    cleaned_text = ' '.join(tokens)
    
    return cleaned_text

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

        # Preprocess the image
        # preprocessed_image_path = preprocess_image(temp_image_path)
        img_cv = cv2.imread(temp_image_path)

        # Now use the path for processing
        img_class = detector.detect(temp_image_path)
        if img_class == "id":
            seg_img = segmenter.segment(temp_image_path)
            if not seg_img:
                raise HTTPException(status_code=400, detail="No regions detected in the image.")
            results = []
            custom_config = r'--oem 1 --psm 6 -c tessedit_char_blacklist=0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'

            for i, seg in enumerate(seg_img):
                
                text = pytesseract.image_to_string(seg, lang='ara+ara_number',config= custom_config )
                cleaned_text = post_process_text(text)
                results.append(cleaned_text)
            return {"id informations": results}
        else:
            return {"detail": "This image is not an ID"}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")