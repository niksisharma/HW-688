import streamlit as st

hw1_page = st.Page("HWs/HW1.py", title="HW 1", icon="🖥️")
hw2_page = st.Page("HWs/HW2.py", title="HW 2", icon="🖥️")
hw3_page = st.Page("HWs/HW3.py", title="HW 3", icon="🖥️")
hw4_page = st.Page("HWs/HW4.py", title="HW 4", icon="🖥️", default=True)

pg = st.navigation([hw1_page, hw2_page, hw3_page, hw4_page])
st.set_page_config(page_title="HW Manager", page_icon="📄")
pg.run()