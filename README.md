# -The-Clarity-Compass-
– A tool that aims to be more of a guide or mentor for students, helping them find direction in their study habits. This project is an innovative tool designed to help students by analyzing their handwritten notes. After a student scans their notes, the application uses AI to provide crucial feedback on three key areas: focus, clarity and content..

AI-Powered Student Note Analyzer
An intelligent web application designed to analyze and score handwritten student notes. This tool uses Optical Character Recognition (OCR) to extract text from an uploaded image and then leverages a semantic AI model to compare the content against a provided reference text.

Features
Image Upload: Easily upload images of handwritten or printed notes.

Reference Text Input: Provide a "golden standard" or reference text for comparison.

AI-Powered Analysis: The application generates two key metrics:

Clarity Score: Rates the legibility and clarity of the notes based on OCR confidence.

Content Score: Rates how semantically similar the student's notes are to the reference text.

Extracted Text Display: Shows the text that was extracted from the image via OCR.

Simple Web Interface: Built with Flask for easy interaction.

How It Works
The application follows a multi-step process to analyze the notes:

Image Pre-processing: The uploaded image is converted to grayscale and an adaptive threshold is applied using OpenCV to improve the quality for OCR.

Text Extraction (OCR): Tesseract OCR is used to scan the processed image and extract all recognizable text.

Clarity Analysis: The confidence level of each word detected by Tesseract is averaged to generate a "Clarity Score." A higher score indicates clearer, more legible text.

Content Analysis: The extracted text and the user-provided reference text are converted into numerical vectors (embeddings) using the Sentence-Transformers (all-MiniLM-L6-v2) model. The cosine similarity between these vectors is calculated to produce a "Content Score," indicating how closely the topics match.

Results: The scores and the extracted text are sent back to the user's browser.

Installation & Setup
Follow these steps to run the project on your local machine.

Prerequisites
Python 3.7+

Tesseract OCR Engine: You must have Tesseract installed on your system. You can download it from the Tesseract at UB Mannheim page.

1. Clone the Repository
git clone https://github.com/your-username/your-repository-name.git
cd your-repository-name

2. Create a Virtual Environment (Recommended)
# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

3. Install Dependencies
Create a requirements.txt file with the following content:

Flask
opencv-python
pytesseract
sentence-transformers
torch
Pillow

Then, install the packages using pip:

pip install -r requirements.txt

4. Configure Tesseract Path
Open the app.py file and update the following line to match the location of your Tesseract installation:

# Update this path if Tesseract is installed elsewhere on your system
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

5. Run the Application
python app.py

The application will start and be accessible at http://127.0.0.1:5000.

Usage
Open your web browser and navigate to http://127.0.0.1:5000.

Click "Choose File" to upload an image of the notes you want to analyze.

Paste the reference text into the text area.

Click the "Analyze Notes" button.

The results, including the Clarity Score, Content Score, and extracted text, will be displayed on the page.

License
This project is licensed under the MIT License. See the LICENSE file for more details.
