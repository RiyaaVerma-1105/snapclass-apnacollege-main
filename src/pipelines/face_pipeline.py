import dlib
import numpy as np
import face_recognition_models
import streamlit as st
import json
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

def load_embedding(embedding_data):
    if embedding_data is None:
        return None
    if isinstance(embedding_data, str):
        try:
            embedding_data = json.loads(embedding_data)
        except:
            pass
    if isinstance(embedding_data, list):
        return np.array(embedding_data)
    return embedding_data

def predict_attendance(class_image_np):
    st.cache_resource.clear()
    
    # STEP 1: Photo se face detect karo
    encodings = get_face_embeddings(class_image_np)
    
    if len(encodings) == 0:
        return {'unknown': True}, [], 0
    
    # STEP 2: Database se sabhi students load karo
    all_students = get_all_students()
    
    if not all_students:
        return {'unknown': True}, [], len(encodings)
    
    detected_student = {}
    
    # STEP 3: HAR EK student se compare karo
    for encoding in encodings:
        best_match_id = None
        best_distance = 999.0
        
        for student in all_students:
            student_id = student.get('student_id')
            name = student.get('name')
            embedding = load_embedding(student.get('face_embedding'))
            
            if embedding is not None:
                # Direct euclidean distance
                distance = np.linalg.norm(embedding - encoding)
                
                # DEBUG print
                print(f"Student ID {student_id} ({name}): Distance = {distance:.4f}")
                
                # Agar distance sabse kam ho aur threshold se kam ho
                if distance < best_distance:
                    best_distance = distance
                    best_match_id = student_id
        
        print(f"Best match: ID {best_match_id}, Distance {best_distance:.4f}")
        
        # === FIX: Threshold 0.75 se ghatakar STRICT 0.45 kiya taaki naye user login na ho payen ===
        if best_match_id is not None and best_distance < 0.45:
            detected_student[best_match_id] = True
            print(f"✅ MATCH FOUND! Student ID: {best_match_id}")
        else:
            detected_student['unknown'] = True
            print(f"❌ NO MATCH! Distance too high: {best_distance}")
    
    student_ids = list(detected_student.keys())
    return detected_student, student_ids, len(encodings)

def train_classifier():
    st.cache_resource.clear()
    return True