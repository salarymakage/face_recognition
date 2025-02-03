from django.http import StreamingHttpResponse, JsonResponse
from django.shortcuts import render, redirect
import cv2
import sqlite3
import logging
from .models import Student
from django.views.decorators.csrf import csrf_exempt

logger = logging.getLogger(__name__)

camera = None
detected_student = None

# Initialize SQLite database if needed
def initialize_database():
    conn = sqlite3.connect("new_students.db")
    conn.execute("""
    CREATE TABLE IF NOT EXISTS STUDENTS (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        FirstName TEXT,
        LastName TEXT,
        Gender TEXT,
        MedicalCondition TEXT,
        Address TEXT,
        EmergencyContact INTEGER
    )
    """)
    conn.close()

initialize_database()

# Class to manage video capture and face recognition
class VideoCamera:
    def __init__(self):
        self.video = cv2.VideoCapture(0)  # Open the camera
        if not self.video.isOpened():
            logger.error("Failed to open camera")
        self.facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        
        # Load pre-trained recognizer model
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        try:
            self.recognizer.read('trainer.yml')  # Change path if necessary
        except cv2.error as e:
            logger.error(f"Error loading recognizer: {e}")
        
    def __del__(self):
        self.video.release()  # Release the video capture resource when done
    
    def get_frame(self):
        success, img = self.video.read()
        if not success:
            logger.error("Failed to capture frame")
            return None

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.facedetect.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            id, conf = self.recognizer.predict(gray[y:y + h, x:x + w])
            
            # Confidence threshold to filter out unknown faces
            if conf < 100:  # Adjust the threshold as needed
                profile = self.getprofile(id)
                if profile:
                    global detected_student
                    detected_student = profile
                    # text = f"ID: {profile[0]} - Name: {profile[1]} {profile[2]}"
                    # cv2.putText(img, text, (x, y + h + 20), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 2)
            else:
                cv2.putText(img, "Unknown", (x, y + h + 20), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 0, 0), 2)

        ret, jpeg = cv2.imencode('.jpg', img)
        if not ret:
            logger.error("Failed to encode frame to JPEG")
            return None
        
        return jpeg.tobytes()

    @staticmethod
    def getprofile(id):
        try:
            conn = sqlite3.connect("new_students.db")
            cursor = conn.execute("SELECT * FROM STUDENTS WHERE id=?", (id,))
            profile = cursor.fetchone()
            if profile:
                # Construct the image path based on the student's ID
                image_path = f'img/{id}.jpg'
                return profile + (image_path,)  # Add the image path to the profile data
            return None
        finally:
            conn.close()  
            
# Function to generate video frames for the camera feed
def gen(camera):
    while True:
        frame = camera.get_frame()
        if frame is None:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

# View to handle video streaming
def video_feed(request):
    global camera
    if camera is None:
        camera = VideoCamera()
    return StreamingHttpResponse(gen(camera), content_type='multipart/x-mixed-replace; boundary=frame')

# View to stop the camera feed
def stop_camera(request):
    global camera
    if camera is not None:
        del camera
        camera = None
    return redirect('index')

# View to display the index page with students data
def index(request):
    global detected_student
    # Fetch students from SQLite database
    conn = sqlite3.connect("new_students.db")
    cursor = conn.execute("SELECT id, FirstName, LastName, Gender, MedicalCondition, Address, EmergencyContact FROM STUDENTS")
    sqlite_students = cursor.fetchall()
    conn.close()

    # Fetch students from Django's database
    django_students = Student.objects.all().values_list('id', 'FirstName', 'LastName', 'Gender', 'MedicalCondition', 'Address', 'EmergencyContact')

    # Combine both lists
    combined_students = list(sqlite_students) + list(django_students)

    # Remove duplicates (based on unique ID)
    unique_students = {student[0]: student for student in combined_students}
    combined_students = list(unique_students.values())

    context = {
        'detected_student': detected_student,
        'students': combined_students,
    }

    if detected_student:
        context['image_path'] = detected_student[-1]  # Pass the image path from the detected student

    return render(request, 'detection/index.html', context)

# API view to return the detected student info as JSON
def detected_student_info(request):
    global detected_student
    if detected_student:
        return JsonResponse({
            'id': detected_student[0],
            'first_name': detected_student[1],
            'last_name': detected_student[2],
            'gender': detected_student[3],
            'medical_condition': detected_student[4],
            'address': detected_student[5],
            'emergency_contact': detected_student[6],
        })
    return JsonResponse({'Error': 'No student detected'}, status=404)

# View to handle student training (adding new student info)
@csrf_exempt
def train(request):
    if request.method == 'POST':
        first_name = request.POST.get('fname')
        last_name = request.POST.get('lname')
        gender = request.POST.get('gender')
        medical_condition = request.POST.get('medical-condition')
        address = request.POST.get('address')
        emergency_contact = request.POST.get('emergency-contact')

        # Save to Django's database
        student = Student(
            FirstName=first_name,
            LastName=last_name,
            Gender=gender,
            MedicalCondition=medical_condition,
            Address=address,
            EmergencyContact=emergency_contact
        )
        student.save()

        # Save to SQLite database
        conn = sqlite3.connect("train.db")
        cursor = conn.cursor()
        cursor.execute("""
        INSERT INTO STUDENTS (FirstName, LastName, Gender, MedicalCondition, Address, EmergencyContact) 
        VALUES (?, ?, ?, ?, ?, ?)
        """, (first_name, last_name, gender, medical_condition, address, emergency_contact))
        conn.commit()
        conn.close()

        return redirect('index')  # Redirect to home (index) page after submission

    return render(request, 'detection/index.html')


def student_details(request, student_id):
    try:
        student = Student.objects.get(pk=student_id)
        response_data = {
            'id': student.id,
            'first_name': student.FirstName,
            'last_name': student.LastName,
            'gender': student.Gender,
            'medical_condition': student.MedicalCondition,
            'address': student.Address,
            'emergency_contact': student.EmergencyContact
        }
        return JsonResponse(response_data)
    except Student.DoesNotExist:
        return JsonResponse({'Error': 'Student not found'}, status=404)
    
import os
import cv2
import numpy as np
from django.shortcuts import render, redirect
from django.http import JsonResponse
from .models import User  # Make sure your User model is defined
from cv2 import face

# View to handle user data, image capture, and model training
def submit_data(request):
    if request.method == 'POST':
        # Retrieve form data
        fname = request.POST.get('fname')
        lname = request.POST.get('lname')
        gender = request.POST.get('gender')
        medical_condition = request.POST.get('medical-condition')
        address = request.POST.get('address')
        emergency_contact = request.POST.get('emergency-contact')

        # Save user data in the database
        user = User(
            first_name=fname,
            last_name=lname,
            gender=gender,
            medical_condition=medical_condition,
            address=address,
            emergency_contact=emergency_contact
        )
        user.save()

        # Create a unique folder name based on user first name and last name
        folder_name = f"{fname}_{lname}"
        dataset_path = os.path.join('dataset', folder_name)

        if not os.path.exists(dataset_path):
            os.makedirs(dataset_path)

        # Capture images using webcam and store them
        capture_images(user.user_id, dataset_path)

        # Train a new model for the user
        train_model(dataset_path, folder_name)

        return JsonResponse({'status': 'success', 'message': f'Images captured and model trained for {fname} {lname}.'})

    return JsonResponse({'status': 'error', 'message': 'Invalid request method.'})


# Function to capture images from webcam and store them
def capture_images(user_id, dataset_path):
    # Initialize face detector
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Initialize the webcam
    cam = cv2.VideoCapture(0)
    cam.set(3, 640)  # Set video width
    cam.set(4, 480)  # Set video height

    print(f"\n [INFO] Initializing face capture for User ID: {user_id}. Look at the camera...")
    count = 0

    # Start capturing faces
    while True:
        ret, img = cam.read()
        if not ret:
            break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            count += 1

            # Save the captured image into the folder
            cv2.imwrite(f"{dataset_path}/User.{user_id}.{count}.jpg", gray[y:y + h, x:x + w])

            cv2.imshow('image', img)

        k = cv2.waitKey(100) & 0xff  # Press 'ESC' to exit
        if k == 27:  # ESC key pressed
            break
        elif count >= 60:  # Capture 60 images and then stop
            break

    cam.release()
    cv2.destroyAllWindows()

    print(f"\n [INFO] {count} images captured for User ID: {user_id}")


# Function to train a face recognition model based on captured images
def train_model(dataset_path, folder_name):
    # Initialize the recognizer
    recognizer = face.LBPHFaceRecognizer_create()
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    print(f"\n [INFO] Training model for {folder_name}")
    faces, ids = get_images_and_labels(dataset_path, face_detector)
    recognizer.train(faces, np.array(ids))

    # Save the trained model into a separate file
    model_file = f'trainer_{folder_name}.yml'
    recognizer.save(model_file)
    print(f"\n [INFO] Model trained and saved as {model_file}")


# Helper function to get images and label data from the folder
def get_images_and_labels(path, face_detector):
    image_paths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]
    face_samples = []
    ids = []

    for image_path in image_paths:
        img_numpy = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        id = int(os.path.split(image_path)[-1].split(".")[1])
        faces = face_detector.detectMultiScale(img_numpy)

        for (x, y, w, h) in faces:
            face_samples.append(img_numpy[y:y + h, x:x + w])
            ids.append(id)

    return face_samples, ids
