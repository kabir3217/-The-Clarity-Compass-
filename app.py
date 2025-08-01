import os
from flask import Flask, request, render_template, jsonify
import cv2
import pytesseract
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

print("Loading semantic analysis model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded.")


def analyze_student_notes(image_path: str, reference_text: str):
   
    try:
       
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        processed_img = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        ocr_data = pytesseract.image_to_data(processed_img, output_type=pytesseract.Output.DICT)
        
        word_confidences = [int(conf) for conf in ocr_data['conf'] if int(conf) > 0]
        full_text = " ".join([text for i, text in enumerate(ocr_data['text']) if int(ocr_data['conf'][i]) > 0])
        
        if not word_confidences:
            return {"error": "OCR could not detect any text. Please use a clearer image."}

        average_confidence = sum(word_confidences) / len(word_confidences)
        focus_clarity_score = (average_confidence / 100) * 10
        
        embedding_student = model.encode(full_text, convert_to_tensor=True)
        embedding_reference = model.encode(reference_text, convert_to_tensor=True)
        cosine_similarity = util.cos_sim(embedding_student, embedding_reference)
        content_style_score = cosine_similarity.item() * 10
        

        return {
            "focus_clarity_score": f"{focus_clarity_score:.1f}",
            "content_style_score": f"{content_style_score:.1f}",
            "ocr_extracted_text": full_text
        }

    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"}


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'noteImage' not in request.files or 'referenceText' not in request.form:
        return jsonify({"error": "Missing image or reference text."}), 400

    file = request.files['noteImage']
    reference_text = request.form['referenceText']

    if file.filename == '':
        return jsonify({"error": "No image selected."}), 400

    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        results = analyze_student_notes(filepath, reference_text)
        
        os.remove(filepath)
        
        return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True, port=5000)