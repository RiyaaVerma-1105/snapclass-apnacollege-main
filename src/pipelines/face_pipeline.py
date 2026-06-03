import dlib
import numpy as np
import face_recognition_models
from sklearn.svm import SVC
import streamlit as st

from src.database.db import get_all_students


@st.cache_resource
def load_dlib_models():
    detector = dlib.get_frontal_face_detector() 
    sp = dlib.shape_predictor(face_recognition_models.pose_predictor_model_location())
    facerec = dlib.face_recognition_model_v1(face_recognition_models.face_recognition_model_location())
    return detector, sp, facerec

def get_face_embeddings(image_np):
    detector, sp, facerec = load_dlib_models()
    faces = detector(image_np, 1)
    encodings = []
    for face in faces:
        shape = sp(image_np, face)
        face_descriptor = facerec.compute_face_descriptor(image_np, shape, 1)
        encodings.append(np.array(face_descriptor))
    return encodings

@st.cache_resource
def get_trained_model():
    X, y = [], []
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
    
    clf = SVC(kernel='linear', probability=True, class_weight='balanced')
    try:
        if len(set(y)) >= 2:
            clf.fit(X, y)
    except ValueError:
        pass

    return {'clf': clf, 'X': X, "y": y}

def train_classifier():
    st.cache_resource.clear()
    model_data = get_trained_model()
    return bool(model_data)

def predict_attendance(class_image_np):
    # CRITICAL: Har prediction par purana ziddi server cache clear karo
    st.cache_resource.clear()
    
    encodings = get_face_embeddings(class_image_np)
    detected_student = {}
    model_data = get_trained_model()

    if not model_data:
        if len(encodings) > 0:
            detected_student['unknown'] = True
        return detected_student, [], len(encodings)
    
    clf = model_data['clf']
    X_train = model_data['X']
    y_train = model_data['y']
    all_students = sorted(list(set(y_train)))

    for encoding in encodings:
        predicted_id = None
        is_confident = False

        # FIX 1: Agar 2 ya usse zyada students hain tabhi SVM prediction chalegi
        if len(all_students) >= 2 and hasattr(clf, "classes_"):
            probs = clf.predict_proba([encoding])[0]
            max_prob_idx = np.argmax(probs)
            if probs[max_prob_idx] >= 0.80:  # Strict 80% confidence boundary
                predicted_id = int(clf.classes_[max_prob_idx])
                is_confident = True
        
        # FIX 2: Agar database me sirf 1 hi student hai, toh blindly true nahi karenge
        elif len(all_students) == 1:
            predicted_id = int(all_students[0])
            # Is single student ke embedding se aane wale face ka distance check karo
            distances = [np.linalg.norm(np.array(emb) - encoding) for t_idx, emb in enumerate(X_train) if y_train[t_idx] == predicted_id]
            min_dist = min(distances) if distances else 999.0
            
            # Agar distance 0.45 se kam hai tabhi bolo ki haan ye wahi bacha hai
            if min_dist <= 0.45:
                is_confident = True
            else:
                is_confident = False  # Warna unknown bna do!

        # FINAL VERIFICATION LAYER
        if predicted_id is not None and is_confident:
            distances = [np.linalg.norm(np.array(emb) - encoding) for t_idx, emb in enumerate(X_train) if y_train[t_idx] == predicted_id]
            min_dist = min(distances) if distances else 999.0

            if min_dist <= 0.45:
                detected_student[predicted_id] = True
            else:
                detected_student['unknown'] = True
        else:
            detected_student['unknown'] = True

    return detected_student, all_students, len(encodings)