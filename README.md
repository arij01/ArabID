# ID Card Recognition Based on Arabic OCR System

## Overview

This project focuses on extracting and processing text from Egyptian and Marrocan ID cards, specifically targeting Arabic OCR (Optical Character Recognition). The system uses deep learning models to detect and segment ID cards, and OCR technology to recognize text within the segments.

## Features

- **ID Detection:** Identifies whether the uploaded image contains an ID card.
- **Segmentation:** Segments the ID card into different regions (e.g., name, number, etc.).
- **Text Extraction:** Uses OCR to extract text from each segmented region.
- **Custom Text Processing:** Cleans and filters the extracted text to enhance readability.
- **Ordering of Text Segments:** Processes text segments in a specific, user-defined order.

## Installation

To run this project, you'll need to have Python 3.8 or higher installed. Additionally, you will need to install the following dependencies:

1. **FastAPI:** For creating the API.
2. **Pillow:** For image processing.
3. **PyTesseract:** For OCR functionality.
4. **NLTK:** For text processing.
5. **Tesseract-OCR:** The OCR engine.

You can install the required Python libraries using pip:

```bash
pip install fastapi uvicorn pillow pytesseract nltk
```
Tesseract-OCR needs to be installed separately. You can download and install it from [Tesseract's official repository](https://github.com/tesseract-ocr/tesseract).
## Configuration

- **Models:**
  - `classification.pt`: Model for detecting ID cards.
  - `segmentation.pt`: Model for segmenting the ID card into regions.

- **Tesseract Configuration:**
  - Ensure that `pytesseract.pytesseract.tesseract_cmd` points to the path where Tesseract-OCR is installed.
## Usage

1. **Start the API Server**

   To run the FastAPI server, execute the following command in your terminal:

   ```bash
   uvicorn main:app --reload
   ```
 This command starts the server and enables automatic reloading for development.
 2.  **Upload an Image:**
 You can send a POST request to the /predict/ endpoint with an image file. You can use tools like curl, Postman, or create a simple frontend to interact with the API.
### Example Image
![Votre-texte-de-paragraphe-43-_png_jpg rf 7d7d26bf750d2026ca166273c800eceb](https://github.com/user-attachments/assets/33648fef-08fe-49f5-9e74-3d532e53f318)
### Example Output
The API will process the image and return the extracted text from the ID card in the specified order. The response will be in JSON format.
 ```bash
{
    "id informations": [
        "أحمد",
        "محمد إبراهيم علي",
        "منطقة الليدو",
        "المنتزه",
        "الإسكندرية",
        "QW5613784"
    ]
}
   ```
