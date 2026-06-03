import os
from supabase import create_client, Client

# Streamlit Cloud (Production)
try:
    import streamlit as st
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
except:
    # LOCAL DEVELOPMENT - Apne credentials yahan hardcoded
    SUPABASE_URL = "https://oennheztsulqxtjlaiby.supabase.co"
    SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im9lbm5oZXp0c3VscXh0amxhaWJ5Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzkzNzk2MzMsImV4cCI6MjA5NDk1NTYzM30.VIjWJj89hPpZuTvXx2gPUmpyyUCIUPQZVZcvpPozZWM"

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)