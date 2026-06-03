import streamlit as st

from src.ui.base_layout import style_background_dashboard, style_base_layout

from src.components.header import header_dashboard
from src.components.footer import footer_dashboard
from PIL import Image
import numpy as np
from src.pipelines.face_pipeline import predict_attendance, get_face_embeddings, train_classifier
from src.pipelines.voice_pipeline import get_voice_embedding
from src.database.db import get_all_students, create_student, get_student_subjects, get_student_attendance, unenroll_student_to_subject
import time

from src.components.dialog_enroll import enroll_dialog
from src.components.subject_card import subject_card

def student_dashboard():
    student_data = st.session_state.student_data
    student_id = student_data['student_id']
    c1, c2 = st.columns(2, vertical_alignment='center', gap='xxlarge')
    with c1:
        header_dashboard()
    with c2:
        st.subheader(f"""Welcome, {student_data['name']} """)
        if st.button("Logout", type='secondary', key='loginbackbtn', shortcut="control+backspace"):
            st.session_state['is_logged_in'] = False
            del st.session_state.student_data 
            st.rerun()


    st.space()

    c1, c2 =st.columns(2)
    with c1:
        st.header('Your Enrolled Subjects')
    with c2:
        if st.button('Enroll in Subject', type='primary', width='stretch'):
            enroll_dialog()


    st.divider()


    with st.spinner('Loading your enrolled subjects..'):
        subjects = get_student_subjects(student_id)
        logs = get_student_attendance(student_id)

    stats_map = {}

    for log in logs:
        sid = log['subject_id']

        if sid not in stats_map:
            stats_map[sid] = {"total":0, "attended": 0}

        stats_map[sid]['total'] +=1

        if log.get('is_present'):
            stats_map[sid]['attended'] += 1


    cols = st.columns(2)
    for i, sub_node in enumerate(subjects):
        sub = sub_node['subjects']
        sid = sub['subject_id']


        stats = stats_map.get(sid,{"total":0, "attended": 0} )
        def unenroll_button():
                if st.button("Unenroll from tihs course", type='tertiary', width='stretch', icon=':material/delete_forever:'):
                    unenroll_student_to_subject(student_id, sid)
                    st.toast(f'Unenrolled from {sub["name"]} successfully!')
                    #st.toast(f"Unenrolled from {sub['name']} successfully!")
                    st.rerun()

        with cols[i % 2]:

            subject_card(
                name = sub['name'],
                code =sub['subject_code'],
                section = sub['section'],
                stats = [
                    ('📅', 'Total', stats['total']),
                    ('✅', 'Attended', stats['attended']),
                ],
                footer_callback=unenroll_button
            )
    footer_dashboard()


def student_screen():
    style_background_dashboard()
    style_base_layout()

    if "student_data" in st.session_state:
        student_dashboard()
        return
    
    c1, c2 = st.columns(2, vertical_alignment='center', gap='xxlarge')
    with c1:
        header_dashboard()
    with c2:
        if st.button("Go back to Home", type='secondary', key='loginbackbtn', shortcut="control+backspace"):
            st.session_state['login_type'] = None
            st.rerun()

    st.header('Login using FaceID', text_alignment='center')
    st.space()
    st.space()
    
    # 1. Registration state ko persist rakhne ke liye Session State use karein
    if "show_registration" not in st.session_state:
        st.session_state.show_registration = False

    photo_source = st.camera_input("Position your face in the center")

    # Agar abhi tak registration form open nahi hua hai, tabhi AI scan chalega
    if photo_source and not st.session_state.show_registration:
        img = np.array(Image.open(photo_source))

        with st.spinner('AI is scanning..'):
            detected, all_ids, num_faces = predict_attendance(img)

            if num_faces == 0:
                st.warning('Face not found!')
            elif num_faces > 1:
                st.warning('Multiple faces found')
            else:
                if detected:
                    student_id = list(detected.keys())[0]
                    all_students = get_all_students()
                    student = next((s for s in all_students if s['student_id'] == student_id), None)

                    if student:
                        st.session_state.is_logged_in = True
                        st.session_state.user_role = 'student'
                        st.session_state.student_data = student
                        st.toast(f"Welcome Back {student['name']}")
                        time.sleep(1)
                        st.rerun()
                else:
                    st.info('Face not recognized! You might be a new student!')
                    st.session_state.show_registration = True
                    st.rerun()  # State update karke rerun karenge taaki form turant dikhe

    # 2. Registration Form Logic (Agar face recognize nahi hua)
    if st.session_state.show_registration:
        with st.container(border=True):
            st.header('Register new Profile')
            
            # Streamlit form ka use karenge taaki input daalte hi camera reset na ho
            with st.form("new_student_register_form"):
                new_name = st.text_input("Enter your name", placeholder='E.g. Riya Verma')

                st.subheader('Optional : Voice Enrollment')
                st.info("Enroll your for voice only attendance")

                audio_data = None
                try:
                    audio_data = st.audio_input('Record a short phrase like I am present, My name is Akash.')
                except Exception:
                    st.error('Audio Data failed!')

                # Form ka submit button
                submit_registration = st.form_submit_button('Create Account', type='primary')
                
                # Agar user form cancel karke firse photo lena chahe
                cancel_registration = st.form_submit_button('Try Face Login Again', type='secondary')

            if cancel_registration:
                st.session_state.show_registration = False
                st.rerun()

            if submit_registration:
                if new_name:
                    if photo_source:  # Ensure image is present
                        with st.spinner('Creating profile..'):
                            img = np.array(Image.open(photo_source))
                            encodings = get_face_embeddings(img)
                            if encodings:
                                face_emb = encodings[0].tolist()

                                voice_emb = None
                                if audio_data:
                                    voice_emb = get_voice_embedding(audio_data.read())

                                response_data = create_student(new_name, face_embedding=face_emb, voice_embedding=voice_emb)

                                if response_data:
                                    train_classifier()
                                    st.session_state.is_logged_in = True
                                    st.session_state.user_role = 'student'
                                    st.session_state.student_data = response_data[0]
                                    st.session_state.show_registration = False  # Reset flag
                                    st.toast(f'Profile Created! Hi {new_name}!')
                                    time.sleep(1)
                                    st.rerun()
                            else:
                                st.error('Couldnt capture your facial features for registration. Please reposition your face.')
                    else:
                        st.error('Please take a photo first!')
                else:
                    st.warning('Please enter your name!')

    footer_dashboard()