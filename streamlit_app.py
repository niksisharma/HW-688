import streamlit as st

hw1_page = st.Page("HWs/HW1.py", title="HW 1", icon="🖥️")
hw2_page = st.Page("HWs/HW2.py", title="HW 2", icon="🖥️", default=True)

pg = st.navigation([hw1_page, hw2_page])
st.set_page_config(page_title="HW Manager", page_icon="📄")
pg.run()
