import cv2
import pytesseract
from PIL import Image
from sentence_transformers import SentenceTransformer, util

try:
    pytesseract.get_tesseract_version()
except pytesseract.TesseractNotFoundError:
    print("Tesseract not found. Please install it or set the path in the script.")
    exit()


def analyze_student_notes(image_path: str, reference_text: str):
    
    try:
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        processed_img = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        

    except Exception as e:
        return {"error": f"Could not read or process image. Error: {e}"}

    
    try:
        ocr_data = pytesseract.image_to_data(processed_img, output_type=pytesseract.Output.DICT)
    except Exception as e:
        return {"error": f"OCR processing failed. Error: {e}"}

    word_confidences = []
    full_text = []
    n_boxes = len(ocr_data['level'])
    for i in range(n_boxes):
        if int(ocr_data['conf'][i]) > 0:
            word_confidences.append(int(ocr_data['conf'][i]))
            full_text.append(ocr_data['text'][i])

    if not word_confidences:
        return {"error": "OCR could not detect any text on the image."}

    average_confidence = sum(word_confidences) / len(word_confidences)
    focus_clarity_score = (average_confidence / 100) * 10
    
    student_text = " ".join(full_text)

    if not student_text.strip():
        return {"error": "OCR did not extract any readable text."}
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    embedding_student = model.encode(student_text, convert_to_tensor=True)
    embedding_reference = model.encode(reference_text, convert_to_tensor=True)
    
    cosine_similarity = util.cos_sim(embedding_student, embedding_reference)
    
    content_style_score = (cosine_similarity.item() * 10)

    results = {
        "focus_clarity_score": f"{focus_clarity_score:.1f}/10",
        "content_style_score": f"{content_style_score:.1f}/10",
        "ocr_extracted_text": student_text,
        "analysis": {
            "clarity_explanation": f"Based on an average OCR confidence of {average_confidence:.1f}%. "
                                   "Higher confidence suggests clearer handwriting.",
            "content_explanation": f"The note's content has a semantic similarity of "
                                   f"{cosine_similarity.item():.2%} to the reference answer."
        }
    }
    
    return results



if __name__ == "__main__":
    student_note_image = "student_note.png"  

    model_answer = """
    Photosynthesis is a process used by plants, algae, and some bacteria to
    convert light energy into chemical energy. This process uses sunlight,
    water, and carbon dioxide to create glucose, which is a sugar that
    provides energy, and oxygen as a byproduct.
    """  # CHANGE THIS to your reference text

    performance_report = analyze_student_notes(student_note_image, model_answer)

    print("--- Student Note Performance Report ---")
    if "error" in performance_report:
        print(f"An error occurred: {performance_report['error']}")
    else:
        print(f"⭐ Focus & Clarity Score: {performance_report['focus_clarity_score']}")
        print(f"📚 Content & Style Score: {performance_report['content_style_score']}")
        print("\n--- Details ---")
        print(f"Clarity: {performance_report['analysis']['clarity_explanation']}")
        print(f"Content: {performance_report['analysis']['content_explanation']}")
        print("\n--- Extracted Text from Note ---")
        print(f"\"{performance_report['ocr_extracted_text']}\"")
        print("------------------------------------")