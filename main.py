from fastapi import FastAPI, File, UploadFile, HTTPException
from util import IDDetector, IDSegmenter,TextPostProcessor
import pytesseract


app = FastAPI()

detector = IDDetector('classification.pt')
segmenter = IDSegmenter('segmentation.pt')
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
post_process_text=TextPostProcessor()
    

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

        # Class names to process in this specific order
        class_order = ["name", "family name", "neighborhood", "city", "state", "number", "Code"]
        
        img_class = detector.detect(temp_image_path)
        if img_class == "id":
            # Get segments and their corresponding class names
            seg_img, class_names = segmenter.segment(temp_image_path)
            if not seg_img:
                raise HTTPException(status_code=400, detail="No regions detected in the image.")

            results = []
            custom_config = r'--oem 3 --psm 11'

            for ordered_class in class_order:
                # Find the segment corresponding to the current class
                for seg, class_name in zip(seg_img, class_names):
                    if class_name == ordered_class:  # Process based on the desired class order
                        if class_name == "number":
                            text = pytesseract.image_to_string(seg, lang='ara_number', config=custom_config)
                        elif class_name == "Code":
                            text = pytesseract.image_to_string(seg, lang='ara_combined', config=custom_config)
                        else:
                            text = pytesseract.image_to_string(seg, lang='ara', config=custom_config)

                        print(f"Extracted text for {class_name}: {text}")
                        cleaned_text = post_process_text.clean_text(text)
                        results.append(cleaned_text)
                        break  # Move to the next class once found

            return {"id informations": results}

        else:
            return {"detail": "This image is not an ID"}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")
