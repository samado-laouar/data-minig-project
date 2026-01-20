import streamlit as st
from config.settings import Config
from pages import data_import, preprocessing, algorithms, results

# Page configuration
st.set_page_config(
    page_title=Config.APP_TITLE,
    page_icon=Config.PAGE_ICON,
    layout=Config.LAYOUT,
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown(f'''
    <style>
    .main {{
        background-color: {Config.BACKGROUND_COLOR};
    }}
    .stButton>button {{
        background-color: {Config.PRIMARY_COLOR};
        color: white;
        border-radius: 8px;
        padding: 10px 24px;
        border: none;
        font-weight: 500;
        font-size: 16px;
    }}
    .stButton>button:hover {{
        background-color: {Config.SECONDARY_COLOR};
    }}
    .nav-button {{
        margin: 20px 0;
    }}
    .step-indicator {{
        background: linear-gradient(90deg, {Config.PRIMARY_COLOR} 0%, #81C784 100%);
        padding: 15px;
        border-radius: 10px;
        color: white;
        text-align: center;
        font-size: 18px;
        font-weight: bold;
        margin-bottom: 20px;
    }}
    .progress-bar {{
        background-color: #e0e0e0;
        border-radius: 10px;
        height: 10px;
        margin: 20px 0;
    }}
    .progress-fill {{
        background-color: {Config.PRIMARY_COLOR};
        height: 100%;
        border-radius: 10px;
        transition: width 0.3s ease;
    }}
    </style>
''', unsafe_allow_html=True)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'preprocessed_data' not in st.session_state:
    st.session_state.preprocessed_data = None
if 'results' not in st.session_state:
    st.session_state.results = {}
if 'current_step' not in st.session_state:
    st.session_state.current_step = 0

# Define steps
STEPS = [
    {"name": "Data Import", "icon": "📁"},
    {"name": "Preprocessing", "icon": "🔧"},
    {"name": "Algorithms", "icon": "🚀"},
    {"name": "Results", "icon": "📊"}
]

# Title
st.title(f"{Config.PAGE_ICON} {Config.APP_TITLE}")

# Progress indicator
current_step = st.session_state.current_step
progress_percentage = ((current_step + 1) / len(STEPS)) * 100

st.markdown(f'''
    <div class="step-indicator">
        Step {current_step + 1} of {len(STEPS)}: {STEPS[current_step]["icon"]} {STEPS[current_step]["name"]}
    </div>
    <div class="progress-bar">
        <div class="progress-fill" style="width: {progress_percentage}%"></div>
    </div>
''', unsafe_allow_html=True)

st.markdown("---")

# Sidebar - Step Overview
with st.sidebar:
    st.header("📋 Steps Overview")
    for idx, step in enumerate(STEPS):
        if idx < current_step:
            st.success(f"{step['icon']} {step['name']} ✓")
        elif idx == current_step:
            st.info(f"➡️ {step['icon']} {step['name']} (Current)")
        else:
            st.write(f"{step['icon']} {step['name']}")
    
    st.markdown("---")
    
    # Reset button
    if st.button("🔄 Start Over"):
        st.session_state.current_step = 0
        st.session_state.data = None
        st.session_state.preprocessed_data = None
        st.session_state.results = {}
        st.rerun()

# Render current step
if current_step == 0:
    data_import.render()
elif current_step == 1:
    preprocessing.render()
elif current_step == 2:
    algorithms.render()
elif current_step == 3:
    results.render()

# Navigation buttons
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])

with col1:
    if current_step > 0:
        if st.button("⬅️ Back", use_container_width=True, key="back_btn"):
            st.session_state.current_step -= 1
            st.rerun()

with col3:
    if current_step < len(STEPS) - 1:
        # Validation before allowing next
        can_proceed = True
        if current_step == 0 and st.session_state.data is None:
            can_proceed = False
        elif current_step == 1 and st.session_state.preprocessed_data is None:
            can_proceed = False
        
        if can_proceed:
            if st.button("Next ➡️", use_container_width=True, key="next_btn"):
                st.session_state.current_step += 1
                st.rerun()
        else:
            st.button("Next ➡️", use_container_width=True, disabled=True, key="next_btn_disabled")
            if current_step == 0:
                st.caption("⚠️ Please upload data first")
            elif current_step == 1:
                st.caption("⚠️ Please preprocess data first")