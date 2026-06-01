import streamlit as st


def footer_home():
    
    st.markdown(f"""
        <div style="margin-top:2rem; display:flex; gap:6px; justify-content:center; align-items:center">
        <p style="font-weight:bold; color:white;"> Created with ❤️ by </p>  
        <p style="
            color:white;
            font-weight:bold;
            font-size:18px;
        ">
            Code Crafter
        </p>

        </div>
        </div>
                
                """, unsafe_allow_html=True)
def footer_dashboard():
        st.markdown(f"""
        <div style="margin-top:2rem; display:flex; gap:6px; justify-content:center; align-items:center">
        <p style="font-weight:bold; color:black;"> Created with ❤️ by </p>  
        <p style="
            color:black;
            font-weight:bold;
            font-size:18px;
        ">
            Code Crafter
        </p>

        </div>
        
                
                """, unsafe_allow_html=True)
        
    
        