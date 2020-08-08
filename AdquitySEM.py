import streamlit as st
import SessionState
from app import main

### Change title and icon of streamlit (not yet in master, will come in the near future)
# st.set_page_config(
# 	layout="centered",  # Can be "centered" or "wide". In the future also "dashboard", etc.
# 	initial_sidebar_state="auto",  # Can be "auto", "expanded", "collapsed"
# 	page_title="Adquity SEM",  # String or None. Strings get appended with "• Streamlit". 
# 	page_icon="/root/.streamlit/adquity_icon.png",  # String, anything supported by st.image, or None.
# )

# To hide streamlit logo and menu
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Password       
USER = 'admin'
PASSWORD = '123'

auth_state = SessionState.get(password='', user='')
if (auth_state.password != PASSWORD) or (auth_state.user != USER):
    user_plaeholder = st.sidebar.empty()
    pwd_placeholder = st.sidebar.empty()
    user = user_plaeholder.text_input("User:", value="")
    pwd = pwd_placeholder.text_input("Password:", value="", type="password")
    auth_state.user = user
    auth_state.password = pwd
    if (auth_state.password == PASSWORD) & (auth_state.user == USER):
        pwd_placeholder.empty()
        user_plaeholder.empty()
        main()
    elif auth_state.password != '':
        st.error("Incorrect user or password")
else:
    main()   