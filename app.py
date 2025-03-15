import os
import io
import cv2
import numpy as np
import base64
from flask import Flask, logging, request, jsonify, send_from_directory
from PIL import Image
from inference_sdk import InferenceHTTPClient
from flask_cors import CORS
import supervision as sv
from dotenv import load_dotenv

load_dotenv()  

app = Flask(__name__)
CORS(app)  

OUTPUT_DIR = 'static'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

CLIENT = InferenceHTTPClient(api_url="https://detect.roboflow.com", api_key=os.getenv("API_KEY"))
MODEL_ID = "oil-palm-44lwi/2"
VIDEO_MODEL_ID = "palm-fruit-ripeness-detection/3"

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image_file = request.files['image']
    
    temp_image_path = "temp_image.jpg"
    image_file.save(temp_image_path)

    image = cv2.imread(temp_image_path)
    
    if image is None:
        logging.error("Failed to read the image with OpenCV.")
        os.remove(temp_image_path)  
        return jsonify({"error": "Failed to read the image."}), 500

    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    temp_cv_image_path = "temp_cv_image.jpg"
    pil_image.save(temp_cv_image_path)

    try:
        result = CLIENT.infer(temp_cv_image_path, model_id=MODEL_ID)
        print(result, "\n\n\n")

        detections = sv.Detections.from_inference(result)

        box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()

        labels = [p['class'] for p in result['predictions']]
        confidences = [p['confidence'] for p in result['predictions']]
        label_and_confidence = [f"{labels[i]}: {confidences[i]:.2f}" for i in range(len(labels))]

        annotated_image = box_annotator.annotate(scene=image, detections=detections)
        annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=label_and_confidence)

        annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        img_result = Image.fromarray(annotated_image_rgb)

        img_byte_arr = io.BytesIO()
        img_result.save(img_byte_arr, format='JPEG')
        img_byte_arr.seek(0)

        img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')

        os.remove(temp_image_path)

        response = {
            "detections": result['predictions'],
            "annotated_image": img_base64
        }

        return jsonify(response)

    except Exception as e:
        os.remove(temp_image_path)  # Clean up temporary image on error
        return jsonify({"error": str(e)}), 500
    

@app.route('/detect-video', methods=['POST'])
def detect_video():
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    video_file = request.files['video']
    print("Received video file:", video_file)
    
    temp_video_path = "temp_video.mp4"
    output_video_path = os.path.join(OUTPUT_DIR, "output_video.mp4")
    video_file.save(temp_video_path)
    
    try:
        cap = cv2.VideoCapture(temp_video_path)
        if not cap.isOpened():
            return jsonify({"error": "Could not open video file."}), 500

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_path = "frame.jpg"
            cv2.imwrite(frame_path, frame)

            try:
                result = CLIENT.infer(frame_path, model_id=VIDEO_MODEL_ID)
                if not result:
                    raise ValueError("No result returned from inference.")
                print(result)
                
            except Exception as infer_error:
                cap.release()
                out.release()
                os.remove(temp_video_path)
                print("Inference error:", infer_error)
                return jsonify({"error": f"Inference error: {str(infer_error)}"}), 500

            detections = sv.Detections.from_inference(result)
            box_annotator = sv.BoxAnnotator()
            label_annotator = sv.LabelAnnotator()
            labels = [f"{p['class']}: {p['confidence']:.2f}" for p in result['predictions']]
            annotated_frame = box_annotator.annotate(scene=frame, detections=detections)
            annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
            out.write(annotated_frame)

        cap.release()
        out.release()
        os.remove(temp_video_path)

        # Return the URL of the output video
        return jsonify({"video_url": f"http://localhost:5000/static/output_video.mp4"})

    except Exception as e:
        if cap.isOpened():
            cap.release()
        if out.isOpened():
            out.release()
        os.remove(temp_video_path)
        if os.path.exists(output_video_path):
            os.remove(output_video_path)
        print("Error in detect_video route:", e)
        return jsonify({"error": f"Server error: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(port=5000, debug=True)
