from src.database.config import supabase
import bcrypt
import streamlit as st

# ==========================================
# 1. PASSWORD HASHING FUNCTIONS
# ==========================================
def hash_pass(pwd):
    return bcrypt.hashpw(pwd.encode(), bcrypt.gensalt()).decode()

def check_pass(pwd, hashed):
    return bcrypt.checkpw(pwd.encode(), hashed.encode())


# ==========================================
# 2. TEACHER DYNAMIC FUNCTIONS
# ==========================================
def check_teacher_exists(username):
    try:
        response = supabase.table("teachers").select("username").eq("username", username).execute()
        return len(response.data) > 0 
    except Exception:
        return False

def create_teacher(username, password, name):
    try:
        data = { "username" : username, "password": hash_pass(password), "name": name}
        response = supabase.table("teachers").insert(data).execute()
        return response.data
    except Exception:
        return None

def teacher_login(username, password):
    try:
        response = supabase.table("teachers").select("*").eq("username", username).execute()
        if response.data:
            teacher = response.data[0]
            if check_pass(password, teacher['password']):
                return teacher
        return None
    except Exception:
        return None

def get_teacher_subjects(teacher_id):
    try:
        response = supabase.table('subjects').select("*, subject_students(count), attendance_logs(timestamp)").eq("teacher_id", teacher_id).execute()
        subjects = response.data

        for sub in subjects:
            sub['total_students'] = sub.get("subject_students", [{}])[0].get('count', 0) if sub.get('subject_students') else 0
            attendance = sub.get('attendance_logs', [])
            unique_sessions = len(set(log['timestamp'] for log in attendance))
            sub['total_classes'] = unique_sessions

            sub.pop('subject_students', None)
            sub.pop('attendance_logs', None)

        return subjects
    except Exception:
        return []

def get_attendance_for_teacher(teacher_id):
    try:
        response = supabase.table('attendance_logs').select("*, subjects!inner(*)").eq('subjects.teacher_id', teacher_id).execute()
        return response.data
    except Exception:
        return []


# ==========================================
# 3. STUDENT DYNAMIC FUNCTIONS
# ==========================================
def get_all_students():
    try:
        # Har baar live database se call hoga, koi local static variable nahi hai
        response = supabase.table('students').select("*").execute()
        return response.data if response.data else []
    except Exception as e:
        st.error(f"Error fetching students: {e}")
        return []

def create_student(new_name, face_embedding=None, voice_embedding=None):
    try:
        data = {'name': new_name, 'face_embedding': face_embedding, "voice_embedding": voice_embedding}
        response = supabase.table('students').insert(data).execute()
        return response.data
    except Exception as e:
        st.error(f"Error creating student: {e}")
        return None

def enroll_student_to_subject(student_id, subject_id):
    try:
        data = {'student_id': student_id, "subject_id": subject_id}
        response = supabase.table('subject_students').insert(data).execute()
        return response.data
    except Exception:
        return None

def unenroll_student_to_subject(student_id, subject_id):
    try:
        response = supabase.table('subject_students').delete().eq('student_id', student_id).eq('subject_id', subject_id).execute()
        return response.data
    except Exception:
        return None

def get_student_subjects(student_id):
    try:
        response = supabase.table('subject_students').select('*, subjects(*)').eq('student_id', student_id).execute()
        return response.data if response.data else []
    except Exception:
        return []

def get_student_attendance(student_id):
    try:
        response = supabase.table('attendance_logs').select('*, subjects(*)').eq('student_id', student_id).execute()
        return response.data if response.data else []
    except Exception:
        return []


# ==========================================
# 4. SUBJECTS & ATTENDANCE CORE
# ==========================================
def create_subject(subject_code, name, section, teacher_id):
    try:
        data = {"subject_code": subject_code, "name": name, "section": section, "teacher_id": teacher_id}
        response = supabase.table("subjects").insert(data).execute()
        return response.data
    except Exception:
        return None

def create_attendance(logs):
    try:
        response = supabase.table('attendance_logs').insert(logs).execute()
        return response.data
    except Exception:
        return None