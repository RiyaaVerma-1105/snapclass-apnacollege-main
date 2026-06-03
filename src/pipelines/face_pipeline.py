import dlib
import numpy as np
import face_recognition_models
from sklearn.svm import SVC
import streamlit as st

from src.database.db import get_all_students


@st.cache_resource
def load_dlib_models():
    detector = dlib.get_frontal_face_detector() 

    sp = dlib.shape_predictor(
        face_recognition_models.pose_predictor_model_location()
    )

    facerec = dlib.face_recognition_model_v1(
        face_recognition_models.face_recognition_model_location()
    )

    return detector, sp, facerec

def get_face_embeddings(image_np):
    detector, sp, facerec = load_dlib_models()
    faces = detector(image_np, 1)

    encodings = []
    for face in faces:
        shape = sp(image_np, face)
        face_descriptor = facerec.compute_face_descriptor(image_np, shape, 1) # 128 embedding
        encodings.append(np.array(face_descriptor))
    return encodings

@st.cache_resource
def get_trained_model():
    X = []
    y = []

    student_db = get_all_students()

    if not student_db:
        return None
    
    for student in student_db:
        embedding = student.get('face_embedding')
        if embedding:
            X.append(np.array(embedding))
            y.append(student.get('student_id'))

    if len(X) == 0:
        return None
    
    # Linear kernel probability toggle enabled
    clf = SVC(kernel='linear', probability=True, class_weight='balanced')

    try:
        clf.fit(X, y)
    except ValueError:
        return None

    return {'clf': clf, 'X': X, "y": y}


def train_classifier():
    st.cache_resource.clear()
    model_data = get_trained_model()
    return bool(model_data)

def predict_attendance(class_image_np):
    encodings = get_face_embeddings(class_image_np)
    detected_student = {}

    model_data = get_trained_model()

    # Agar model trained nahi hai ya database khali hai, to saare faces Unknown hain
    if not model_data or model_data == 0:
        for idx in range(len(encodings)):
            detected_student[f"unknown_{idx}"] = True
        return detected_student, [], len(encodings)
    
    clf = model_data['clf']
    X_train = model_data['X']
    y_train = model_data['y']

    all_students = sorted(list(set(y_train)))
    num_unique_students = len(all_students)

    for idx, encoding in enumerate(encodings):
        predicted_id = None
        is_confident = False

        # LAYER 1: SVM Classifier Protection
        # Agar system me 2 ya usse zyada alag-alag students registered hain, to probability check lagao
        if num_unique_students >= 2:
            probs = clf.predict_proba([encoding])[0]
            max_prob_idx = np.argmax(probs)
            max_prob = probs[max_prob_idx]
            
            # Strict Confidence Threshold: Agar model 75% sure hai tabhi ID consider karein
            if max_prob >= 0.75:
                predicted_id = int(clf.classes_[max_prob_idx])
                is_confident = True
        else:
            # Agar database me sirf 1 hi unique student hai, to model predict_proba nahi kar sakta.
            # Tab hum decision_function ya directly vector math par depend karenge.
            predicted_id = int(all_students[0])
            is_confident = True # Vector distance verification layer ise filter karegi

        # LAYER 2: Vector Distance Verification (Euclidean Distance Validation)
        if predicted_id is not None:
            # Sahi student ke saare registered embeddings se distance compare karein (sirf pehle wale se nahi)
            student_distances = [
                np.linalg.norm(np.array(emb) - encoding)
                for idx_t, emb in enumerate(X_train)
                if y_train[idx_t] == predicted_id
            ]
            
            min_distance = min(student_distances) if student_distances else 999.0
            
            # Strict dlib threshold: 0.55 ya usse kam distance matlab real match.
            # 0.6 thoda loose ho jata hai, 0.53-0.55 optimal hai dlib ke liye.
            if is_confident and min_distance <= 0.55:
                detected_student[predicted_id] = True
            else:
                # Dono me se ek bhi filter fail hua to mark as Unknown
                detected_student[f"unknown_{idx}"] = True
        else:
            detected_student[f"unknown_{idx}"] = True

    return detected_student, all_students, len(encodings)