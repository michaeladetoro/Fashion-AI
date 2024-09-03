from flask import Flask, render_template, request
import cv2
import numpy as np
import mediapipe as mp

app = Flask(__name__)

def calculate_body_height(landmarks, image_height):
    top = min(landmarks[mp.solutions.pose.PoseLandmark.LEFT_EYE.value].y,
              landmarks[mp.solutions.pose.PoseLandmark.RIGHT_EYE.value].y)
    bottom = max(landmarks[mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value].y,
                 landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ANKLE.value].y)
    return (bottom - top) * image_height

def pixels_to_inches(pixels, body_height_pixels, user_height_inches):
    return (pixels / body_height_pixels) * user_height_inches

def process_image(image, user_height_inches, view):
    mp_pose = mp.solutions.pose
    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        
        measurements = {
            'height': user_height_inches,
            'chest': 0,
            'waist': 0,
            'hip': 0,
            'shoulder': 0,
            'sleeve': 0,
            'inseam': 0,
            'outseam': 0
        }
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            image_height, image_width, _ = image.shape
            
            body_height_pixels = calculate_body_height(landmarks, image_height)
            pixels_to_inches_ratio = user_height_inches / body_height_pixels
            
            def get_distance(landmark1, landmark2):
                p1 = np.array([landmark1.x, landmark1.y])
                p2 = np.array([landmark2.x, landmark2.y])
                return np.linalg.norm(p1 - p2) * image_width * pixels_to_inches_ratio
            
            if view == 'front':
                # Chest measurement
                if all(landmarks[i] for i in [mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER]):
                    chest = get_distance(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                         landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value])
                    measurements['chest'] = pixels_to_inches(chest * 2, body_height_pixels, user_height_inches)  # Approximating full chest
                
                # Waist measurement
                if all(landmarks[i] for i in [mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP]):
                    waist = get_distance(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                         landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value])
                    measurements['waist'] = pixels_to_inches(waist * 2, body_height_pixels, user_height_inches)  # Approximating full waist
                
                # Hip measurement
                if all(landmarks[i] for i in [mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP]):
                    hip = get_distance(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                       landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value])
                    measurements['hip'] = pixels_to_inches(hip * 2, body_height_pixels, user_height_inches)  # Approximating full hip
                
                # Inseam and Outseam measurements
                if all(landmarks[i] for i in [mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_ANKLE]):
                    leg_length = get_distance(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                            landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])
                    measurements['inseam'] = pixels_to_inches(leg_length, body_height_pixels, user_height_inches)
                    
                    # Estimate outseam by adding some length for waist height
                    waist_to_hip = get_distance(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]) * 0.2  # Approximate waist position
                    measurements['outseam'] = pixels_to_inches(leg_length + waist_to_hip, body_height_pixels, user_height_inches)
            
            elif view == 'side':
                # Sleeve measurement
                if all(landmarks[i] for i in [mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_WRIST]):
                    measurements['sleeve'] = pixels_to_inches(get_distance(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                                                           landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]), body_height_pixels, user_height_inches)
            
            elif view == 'back':
                # Shoulder measurement
                if all(landmarks[i] for i in [mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER]):
                    measurements['shoulder'] = pixels_to_inches(get_distance(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                                                             landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]), body_height_pixels, user_height_inches)
        
        return measurements

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'front_view' not in request.files or 'side_view' not in request.files or 'back_view' not in request.files:
            return render_template('app.html', error='Please upload all required images')
        
        front_file = request.files['front_view']
        side_file = request.files['side_view']
        back_file = request.files['back_view']
        
        if front_file.filename == '' or side_file.filename == '' or back_file.filename == '':
            return render_template('app.html', error='Please upload all required images')
        
        user_height = request.form.get('height')
        if not user_height:
            return render_template('app.html', error='Please enter your height')
        
        try:
            user_height_inches = float(user_height)
        except ValueError:
            return render_template('app.html', error='Invalid height value')
        
        measurements = {}
        
        for file, view in [(front_file, 'front'), (side_file, 'side'), (back_file, 'back')]:
            # Read the image file
            image_stream = file.read()
            image_array = np.frombuffer(image_stream, np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            
            # Process the image
            view_measurements = process_image(image, user_height_inches, view)
            measurements.update(view_measurements)
        
        return render_template('app.html', measurements=measurements)
    
    return render_template('app.html')

if __name__ == '__main__':
    app.run(debug=True)