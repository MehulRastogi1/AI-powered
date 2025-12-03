import streamlit as st
import joblib
import pandas as pd
import plotly.express as px
from streamlit.components.v1 import html
from textblob import TextBlob
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import sqlite3
from sqdata import *
from datetime import datetime
import numpy as np
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.tree import DecisionTreeRegressor


# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="The AI Revolution Hub",
    layout="wide",
    page_icon="🔍"
)

# Load external CSS
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# ---------- SESSION SETUP ----------
if "module" not in st.session_state:
    st.session_state["module"] = "Home"

protected_pages = ["User Login", "Create User", "AdminLogin","AdminDashboard"]


# =================== SIDEBAR ===================
sbar = st.sidebar

# ---------- TOP BRAND AREA ----------
try:
    sbar.image("images/logo.png", width=230)
except:
    sbar.warning("⚠ logo.png not found in images/ folder!")

sbar.markdown(
    "<div class='sidebar-main-title'>🚀 AI Mega Project</div>",
    unsafe_allow_html=True
)
sbar.markdown(
    "<div class='sidebar-subtitle'>Choose a module below</div>",
    unsafe_allow_html=True
)
sbar.write("---")

# ---------- CHECK LOGIN STATUS ----------
logged_in = st.session_state.get("current_user", None)

# ---------- MASTER MODULE SELECTOR ----------
if logged_in:
    # Show all modules if logged in
    module_options = [
        "Home",
        "Classification",
        "Sentiment Analysis",
        "Movie Recommendation",
        "Linear Regression Models",
        "Time Series Forecasting",
        "NLP Utilities",
        "Audio Processing (NEW)",
        "Image Recognition (NEW)"
    ]
else:
    # Show only Home and Contact if not logged in
    module_options = ["Home", "ABOUT Me"]

module = sbar.selectbox("📂 Select Module", module_options)

sbar.write("---")

# ---------- SUB-MODULES (Dynamic Based on Main Module & Login) ----------
selected_task = None

if logged_in:
    if module == "Classification":
        selected_task = sbar.radio(
            "📌 Classification Tasks",
            ["Spam Detection", "News Detection", "Diabetes Prediction"]
        )

    elif module == "Linear Regression Models":
        selected_task = sbar.radio(
            "📈 Regression Tasks",
            ["Student Performance Prediction", "Car Price Prediction"]
        )

    elif module == "NLP Utilities":
        selected_task = sbar.radio(
            "💬 NLP Tools",
            ["Text Cleaning", "Text Correction", "Keyword Extraction", "Language Detection"]
        )

    elif module == "Audio Processing (NEW)":
        selected_task = sbar.radio(
            "🎵 Audio Tasks",
            ["Speech Recognition", "Emotion from Voice"]
        )

    elif module == "Image Recognition (NEW)":
        selected_task = sbar.radio(
            "🖼 Image Tasks",
            ["Object Detection", "Face Emotion Recognition"]
        )

sbar.write("---")

# ---------- CONTACT ----------
sbar.markdown("### 📞 Developer Contact")
sbar.markdown(
    """
    **Name:** Mehul Rastogi  
    **Email:** mehulrastogi@gmail.com  
    **GitHub:** github.com/username  
    """
)
sbar.write("---")

sbar.markdown(
    "<div class='sidebar-footer'>Made with ❤️ by Mehul • © 2025</div>",
    unsafe_allow_html=True
)

# ---------- UPDATE SESSION ----------
# Update session only if page is not protected
if st.session_state["module"] not in protected_pages:
    st.session_state["module"] = module

    
st.session_state["task"] = selected_task




#===============USER LOGIN==========================
def user_login():
    st.markdown("## 👤 User Login")

    # Back button
    if st.button("⬅ Back to Home"):
        st.session_state["module"] = "Home"
        return

    st.write("")

    username = st.text_input("🧑 Username")
    password = st.text_input("🔑 Password", type="password")

    st.write("")
    login_btn = st.button("Login")

    if login_btn:
        if username.strip() == "" or password.strip() == "":
            st.warning("⚠️ Please enter both username and password.")
        else:
            # ---------------------
            # SQL Login Check
            # ---------------------
            if verify_user(username, password):

                # ✨ Login success
                st.success("✅ Login Successful!")
                st.balloons()

                # Login logs insert karo
                log_id = log_login(username)

                # Save session
                st.session_state["current_user"] = username
                st.session_state["log_id"] = log_id
                st.session_state["module"] = "Home"
            else:
                st.error("❌ Invalid username or password. Try again!")

    st.write("---")

    st.markdown("### ✨ New user? Create an account")

    if st.button("Create Account"):
        st.session_state["module"] = "Create User"
        st.balloons()


#===================CREATE USER========================
def create_user():
    st.markdown("## 🆕 Create User Account")

    # Back button
    if st.button("⬅ Back to Login"):
        st.session_state["module"] = "User Login"
        return

    st.write("")

    name = st.text_input("🧑 Full Name")
    username = st.text_input("📝 Username")
    password = st.text_input("🔐 Password", type="password")

    st.write("")
    create_btn = st.button("Create Account")

    if create_btn:
        if name.strip() == "" or username.strip() == "" or password.strip() == "":
            st.warning("⚠️ All fields are required!")
        else:

            # ----------------------------
            # Add to SQL users table
            # ----------------------------
            success = add_user(username, password)

            if success:
                st.success(f"🎉 Account created successfully for **{name}**!")
                st.balloons()

                # After creation → Go to login screen
                st.session_state["module"] = "User Login"
            else:
                st.error("❌ Username already exists! Try a different one.")




# -------------------------------------
# ADMIN LOGIN PAGE
# -------------------------------------
def admin_login_ui():
    st.title("🔐 Admin Login")

    st.write("Enter your admin credentials:")
    username = st.text_input("Admin Username")
    password = st.text_input("Password", type="password")

    login_btn = st.button("Login")

    if login_btn:
        # 🚨 SQL connect baad me hoga
        if username == "admin" and password == "1234":
            st.success("Login Successful!")
            st.session_state["module"] = "AdminDashboard"
        else:
            st.error("Invalid Credentials")

# -------------------------------------
# ADMIN DASHBOARD (SQL-INTEGRATED)
# -------------------------------------
def admin_dashboard_ui():
    st.title("🛠 Admin Dashboard")

    st.write("Choose what you want to view:")

    col1, col2, col3 = st.columns(3)

    # View USERS button
    with col1:
        if st.button("👤 View Users"):
            st.session_state["admin_view"] = "users"

    # View LOGS button
    with col2:
        if st.button("📜 View Login Logs"):
            st.session_state["admin_view"] = "logs"

    # View CONTACT button
    with col3:
        if st.button("✉️ View Contact Messages"):
            st.session_state["admin_view"] = "contact"

    st.write("---")

    # ======= SQL TABLE OUTPUT =======
    if "admin_view" in st.session_state:

        # ---------- USERS TABLE ----------
        if st.session_state["admin_view"] == "users":
            st.subheader("👤 Users Table")
            try:
                conn = get_connection()
                import pandas as pd
                df = pd.read_sql_query("SELECT * FROM users", conn)
                conn.close()
                st.dataframe(df)
            except Exception as e:
                st.error(f"Error fetching users table: {e}")

        # ---------- LOGS TABLE ----------
        elif st.session_state["admin_view"] == "logs":
            st.subheader("📜 Login / Logout Logs")
            try:
                conn = get_connection()
                import pandas as pd
                df = pd.read_sql_query("SELECT * FROM logs", conn)
                conn.close()
                st.dataframe(df)
            except Exception as e:
                st.error(f"Error fetching logs table: {e}")

        # ---------- CONTACT TABLE ----------
        elif st.session_state["admin_view"] == "contact":
            st.subheader("✉️ Contact Messages")
            try:
                conn = get_connection()
                import pandas as pd
                df = pd.read_sql_query("SELECT * FROM contact", conn)
                conn.close()
                st.dataframe(df)
            except Exception as e:
                st.error(f"Error fetching contact table: {e}")

    # BACK BUTTON
    if st.button("⬅ LOGOUT"):
        st.session_state["module"] = "Home"
        if "admin_view" in st.session_state:
            del st.session_state["admin_view"]

#---------------------------HOME-------------------------------------

def home():

    # ===================== TOP-RIGHT LOGIN BUTTONS ==========================
    c1, c2, c3 = st.columns([6, 1, 1])

    with c2:
        user_btn = st.button("👤 User Login")

    with c3:
        admin_btn = st.button("🛡 Admin Login")

    # Button click → change module
    if user_btn:
        st.session_state["module"] = "User Login"
        

    if admin_btn:
        st.session_state["module"] = "AdminLogin"
        

    with c1:    
        # ---------- HERO SECTION ----------
        st.markdown("""
            <div style="text-align:center; padding:60px 20px;">
                <h1 style="font-size:52px; margin-bottom:10px;">🤖 AI & ML Intelligence Platform</h1>
                <p style="font-size:18px; color:gray; max-width:750px; margin:auto;">
                    A unified platform offering Classification, Regression, Recommendation Systems, NLP Tools,
                    Self-Learning Models, Automated Pipelines, and Intelligent Data Processing — all in one place.
                </p>
            </div>
        """, unsafe_allow_html=True)




    st.markdown("---")

    # ---------- OUR VISION ----------
    st.header("🌍 Our Vision")
    st.write("""
    Our vision is to create a **powerful AI ecosystem** where learners, developers, 
    and businesses can experiment, test, and deploy machine learning models effortlessly.
    
    We aim to simplify ML workflows and make AI accessible for everyone.
    """)

    # ---------- OUR MISSION ----------
    st.header("🎯 Our Mission")
    st.write("""
    - To provide **ready-to-use AI tools** for multiple real-world tasks  
    - To support **Classification, Regression, NLP, and Recommendation models** under one roof  
    - To enable **self-improving models** with continuous data-driven learning  
    - To deliver a clean, modern and interactive interface powered by **Streamlit**  
    """)

    st.markdown("---")

    # ---------- PLATFORM FEATURES ----------
    st.header("🚀 Platform Capabilities")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div style="padding:20px; background:#f5f5f5; border-radius:12px;">
            <h3>📊 Classification Models</h3>
            <p>Binary & Multi-class classification tools for text, numeric and mixed datasets.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="padding:20px; background:#f5f5f5; border-radius:12px;">
            <h3>📈 Regression Engines</h3>
            <p>Predict continuous values using modern regression pipelines and automated tuning.</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div style="padding:20px; background:#f5f5f5; border-radius:12px;">
            <h3>🎬 Recommendation Systems</h3>
            <p>Movie, product, and content recommenders using similarity and deep embeddings.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col4, col5, col6 = st.columns(3)

    with col4:
        st.markdown("""
        <div style="padding:20px; background:#f5f5f5; border-radius:12px;">
            <h3>📝 NLP Utilities</h3>
            <p>Text correction, sentiment analysis, keyword extraction, preprocessing, and more.</p>
        </div>
        """, unsafe_allow_html=True)

    with col5:
        st.markdown("""
        <div style="padding:20px; background:#f5f5f5; border-radius:12px;">
            <h3>🧠 Self-Learning Models</h3>
            <p>Models that update in real-time using continuous data input.</p>
        </div>
        """, unsafe_allow_html=True)

    with col6:
        st.markdown("""
        <div style="padding:20px; background:#f5f5f5; border-radius:12px;">
            <h3>📁 Bulk Processing</h3>
            <p>Upload CSV files, run predictions, retrain models and export outputs efficiently.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ---------- ABOUT THE PLATFORM ----------
    st.header("ℹ️ About This Platform")
    st.write("""
    This AI platform is built using:
    - **Python**  
    - **Streamlit**  
    - **Scikit-learn ML Pipelines**  
    - **Natural Language Processing libraries**  
    - **Recommendation Algorithms**  
    - **Optimized Vectorizers & Cloud-friendly architecture**  

    It is designed to help beginners, students, engineers, analysts, and researchers.
    """)

    st.markdown("---")

    # ---------- CONTACT ----------
    st.header("📞 Contact & Support")
    st.info("""
    **Developer:** Mehul Rasogi    
    **Email:** mehulrastogi@gmail.com  
    **Platform Version:** 1.0.2
    """)

    st.markdown("<br><br>", unsafe_allow_html=True)

    # --- Embed the full contact form here ---
    contact_form_ui()

    # ---------- FOOTER ----------
    st.markdown("""
        <hr>
        <center style="color:gray;">
            © 2025 AI Intelligence Platform • Crafted with ❤️ in Streamlit
        </center>
    """, unsafe_allow_html=True)


#=============================CONTACT ME ==================================
def contact_form_ui():
    st.markdown("## ✉️ Contact Me")
    current_user = st.session_state.get("current_user", "")

    # Optional: Subject / Category
    subject = st.selectbox(
        "📌 Subject / Category",
        ["General", "Bug Report", "Feature Request", "Other"]
    )

    # Input fields
    name = st.text_input("🧑 Full Name")
    username = st.text_input("📝 Username", value=current_user)
    email = st.text_input("📧 Email")
    message = st.text_area("💬 Your Message", height=150, max_chars=500)
    st.caption(f"Characters typed: {len(message)} / 500")

    # Submit button
    if st.button("Send Message 📩"):
        # Validation
        if not name.strip() or not username.strip() or not email.strip() or not message.strip():
            st.warning("⚠️ All fields are required!")
            return

        email_pattern = r"[^@]+@[^@]+\.[^@]+"
        if not re.match(email_pattern, email):
            st.warning("⚠️ Please enter a valid email address.")
            return

        # Prevent double submission
        if st.session_state.get("contact_submitted"):
            st.info("⚠️ You already submitted a message. Please wait before sending another.")
            return

        try:
            # Add to SQL
            add_contact(name, username, email, f"[{subject}]-->{message}")
            
            # Success feedback
            st.success("✅ Your message has been sent successfully! 🎉")
            st.balloons()
            st.info(f"Submitted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

            # Mark as submitted
            st.session_state["contact_submitted"] = True

            # Clear fields
            for key in ["name", "username", "email", "message"]:
                if key in st.session_state:
                    st.session_state[key] = ""

        except Exception as e:
            st.error(f"❌ Something went wrong! {e}")

    # Reset submission flag if needed
    if st.session_state.get("contact_submitted") and st.button("Send Another Message"):
        st.session_state["contact_submitted"] = False


#======================== ABOUT ME ===================================
def about_me_ui():

    # ----------- PAGE HEADER -----------
    st.markdown("""
        <div style="text-align:center; padding:50px;">
            <h1 style="font-size:48px; margin-bottom:5px;">👨‍💻 About the Developer</h1>
            <p style="font-size:20px; color:gray;">Discover the mind behind this AI & ML Platform</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ----------- PROFILE SECTION -----------
    col1, col2 = st.columns([1, 2], gap="large")

    with col1:
        st.image(
            "images/developer.jpg",
            caption="Mehul Rastogi",
            use_container_width=True
        )

    with col2:
        st.markdown(""" <h2>Hi, I'm <b>Mehul Rastogi</b></h2> <p style="font-size:17px; line-height:1.6;"> 
                    I am a passionate learner and developer currently pursuing <b>Data Science</b> from <b>Ducate</b>. <br><br> 
                    I love working on Machine Learning, Artificial Intelligence, and automation-based projects. 
                    This platform is one of my personal projects where I integrate ML models, NLP systems, recommendation engines,
                     and real-world intelligent tools — all in one place. <br><br> My goal is to create impactful AI solutions that help students, developers,
                     and businesses use Machine Learning in the easiest possible way. </p> """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)




    # ----------- INTERESTS SECTION -----------

    st.markdown("""
        <h2 style="text-align:center; margin-bottom:30px;">💬 Interests & Skills</h2>
    """, unsafe_allow_html=True)

    colA, colB, colC = st.columns(3, gap="large")

    card_style = """
        background: linear-gradient(135deg, #ffffff 0%, #f0f3ff 100%);
        padding: 25px;
        border-radius: 18px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    """
    # Hover effect
    hover_script = """
        <style>
            .skill-card:hover {
                transform: translateY(-6px);
                box-shadow: 0 8px 20px rgba(0,0,0,0.15);
            }
        </style>
    """
    st.markdown(hover_script, unsafe_allow_html=True)

    with colA:
        st.markdown(f"""
            <div class="skill-card" style="{card_style}">
                <h2 style="font-size:32px;">📊</h2>
                <h4 style="margin-top:-10px;">Machine Learning</h4>
                <p style="font-size:15px;">Training • Optimization • Pipelines • Deployment</p>
            </div>
        """, unsafe_allow_html=True)

    with colB:
        st.markdown(f"""
            <div class="skill-card" style="{card_style}">
                <h2 style="font-size:32px;">🧠</h2>
                <h4 style="margin-top:-10px;">NLP Systems</h4>
                <p style="font-size:15px;">Sentiment • Text Preprocessing • Tokenization • Tools</p>
            </div>
        """, unsafe_allow_html=True)

    with colC:
        st.markdown(f"""
            <div class="skill-card" style="{card_style}">
                <h2 style="font-size:32px;">🌐</h2>
                <h4 style="margin-top:-10px;">Web Apps</h4>
                <p style="font-size:15px;">Streamlit UI • UX • Automation • Full App Logic</p>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)


    # ----------- CONTACT SECTION -----------

    st.markdown("""
        <h2 style="text-align:center; margin-bottom:10px;">📬 Connect With Me</h2>
        <p style="text-align:center; color:gray;">Feel free to reach out for collaboration or queries.</p>
    """, unsafe_allow_html=True)

    st.info("""
    **Email:** mehulrastogi@gmail.com  
    **LinkedIn:** i don't have  
    **GitHub:** i don't have 
    """)

    st.markdown("<br><br>", unsafe_allow_html=True)


#=====================SPAM_CLASSIFICATION=====================

def run_spam():

    # ---------------- LOAD MODEL ----------------
    model = joblib.load("models/spam_clf_2.pkl")

    # ---------------- MAIN TITLE + ADMIN BUTTON ----------------
    col1, col2 = st.columns([4,1])

    with col1:
        st.markdown("<div class='main-title'>AI-Powered Spam Detection Engine</div>", unsafe_allow_html=True)

    with col2:
        if st.button("Admin Login"):
            st.session_state['show_admin'] = True

    # ---------------- ADMIN LOGIN PAGE ----------------
    if st.session_state.get('show_admin', False):

        st.markdown("### 🔒 Admin Login Page")

        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login", key="login_admin"):
            # Replace below with your secure credentials
            if username == "admin" and password == "1234":
                st.session_state['admin_logged_in'] = True
                st.success("✅ Logged in successfully!")
            else:
                st.error("❌ Invalid credentials")

    # ---------------- ADMIN MODEL UPDATE ----------------
    if st.session_state.get('admin_logged_in', False):
        st.markdown("### ⚡ Update Spam Detection Model")

        file = st.file_uploader("Upload new training data (.csv)", type=["csv"])

        if file:
            df_new = pd.read_csv(file)
            
            
            X_new = df_new.iloc[:,0]
            y_new = df_new.iloc[:,1]

            if st.button("Update Model", key="update_model"):
                # Update the existing model with new data
                model['sgd'].partial_fit(model['hv'].transform(X_new), y_new,classes=['ham','spam'])

                
                # Save updated model
                joblib.dump(model, "spam_clf_2.pkl")
                st.success("✅ Model updated successfully!")

                st.balloons()  

    st.markdown("<div class='sub'>Quick, reliable, and smart spam classification for your messages</div>", unsafe_allow_html=True)

    # ---------------- COLUMNS ----------------
    col1, col2 = st.columns([1, 2], gap='large')



    # ============= LEFT COLUMN (SINGLE MESSAGE) =============
    with col1:

        st.markdown("<div class='left-block'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Single Message Check</div>", unsafe_allow_html=True)

        text = st.text_input("Enter a message")

        # Initialize session states
        if "single_result" not in st.session_state:
            st.session_state.single_result = None
        if "single_proba" not in st.session_state:
            st.session_state.single_proba = None

        # Predict button
        if st.button("Predict"):
            result = model.predict([text])[0]
            proba = model.predict_proba([text])[0]

            st.session_state.single_result = result
            st.session_state.single_proba = proba

            # Display main result
            if result == "spam":
                st.error("🛑 This message is **SPAM**")
            else:
                st.success("✅ This message is **HAM**")

        # -------------------------------
        # MORE INFO
        # -------------------------------
        if st.session_state.single_result is not None:

            if st.button("More Info"):

                spam_prob = st.session_state.single_proba[1] * 100
                ham_prob = st.session_state.single_proba[0] * 100

                if st.session_state.single_result == "spam":
                    prob = spam_prob
                    label = "SPAM"
                else:
                    prob = ham_prob
                    label = "HAM"

                intensity = "Might be" if prob <= 75 else "Sure"

                # ---- PIE CHART ----
                chart_df = pd.DataFrame({
                    "Type": ["Ham", "Spam"],
                    "Confidence": [ham_prob, spam_prob]
                })

                fig = px.pie(
                    chart_df,
                    names="Type",
                    values="Confidence",
                    hole=0.45,
                    color="Type",
                    color_discrete_map={"Ham": "#2DBF73", "Spam": "#FF4B4B"}
                )

                fig.update_layout(
                    showlegend=False,
                    height=240,
                    margin=dict(l=0, r=0, t=10, b=10)
                )

                chart_html = fig.to_html(include_plotlyjs="cdn")

                # ---- FULL CUSTOM BLOCK (NO STREAMLIT SPACING) ----
                html_block = f"""
                <div style="
                    background: linear-gradient(135deg, #F1F5FF, #FFFFFF);
                    padding: 22px 26px;
                    border-radius: 16px;
                    margin-top: 2px;
                    border: 1px solid rgba(120,150,255,0.25);
                    box-shadow: 0 8px 20px rgba(0,0,0,0.08);
                ">
                    <div style="display:flex; gap:25px; align-items:center;">

                        <!-- LEFT INFO -->
                        <div style="
                            flex:1;
                            background:#F0F4FF;
                            padding:18px;
                            border-radius:12px;
                            border-left:6px solid #557CFF;
                            font-size:17px;
                            color:#3d8903;
                            line-height:1.6;
                            font-weight:500;
                            box-shadow:0 2px 7px rgba(0,0,0,0.07);
                        ">
                            📊 <b>Confidence:</b> {prob:.2f}% <br>
                            <b>{intensity} it's {label}</b>
                        </div>

                        <!-- PIE CHART -->
                        <div style="flex:1.2;">
                            {chart_html}
                        </div>

                    </div>
                </div>
                """

                html(html_block, height=420)

        st.markdown("</div>", unsafe_allow_html=True)

    # ============= RIGHT COLUMN (BULK CHECK) =============
    with col2:

        st.markdown("<div class='section-title'>Bulk Message Check</div>", unsafe_allow_html=True)

        # --- File Upload ---
        file = st.file_uploader("Upload your file (.txt or .csv)", type=["txt", "csv"])

        if file:
            if 'bulk_df' not in st.session_state or st.session_state.get('file_changed', True):
                # Read uploaded file once
                st.session_state.bulk_df = pd.read_csv(file, header=None, names=["mgs"], sep='$')
                st.session_state.predicted = False
                st.session_state.show_spam = False
                st.session_state.show_ham = False
                st.session_state.search_keyword = ''
                st.session_state.file_changed = False
            place = st.empty()
            place.dataframe(st.session_state.bulk_df, use_container_width=True)

        # --- Buttons Row ---
        r_col1, r_col2 = st.columns([1,1])
        
        with r_col1:
            if st.button("Predict Bulk", key="b2") and file:
                # Predict and save in session state
                st.session_state.bulk_df['result'] = model.predict(st.session_state.bulk_df["mgs"])
                st.session_state.predicted = True

        # --- Only show summary/filtering if prediction done ---
        if file and st.session_state.predicted:

            # --- Checkboxes Row ---
            c1, c2 = st.columns([1,1])
            with c1:
                st.session_state.show_spam = st.checkbox("🛑 Spam", value=st.session_state.show_spam)
            with c2:
                st.session_state.show_ham = st.checkbox("✅ Ham", value=st.session_state.show_ham)

            # --- Keyword Search with Clear Button ---
            s_col1, s_col2 = st.columns([4,1])
            with s_col1:
                st.session_state.search_keyword = st.text_input(
                    "🔍 Search Keyword", 
                    value=st.session_state.search_keyword
                )
            with s_col2:
                st.text('')    #-------> this is just for gap for push down clear button
                if st.button("❌ Clear", key="clear_search"):
                    st.session_state.search_keyword = ''
                    

            keyword = st.session_state.search_keyword

            # --- Counts ---
            spam_count = (st.session_state.bulk_df['result'] == 'spam').sum()
            ham_count = (st.session_state.bulk_df['result'] == 'ham').sum()
            total = spam_count + ham_count

            spam_perc = spam_count * 100 / total
            ham_perc = ham_count * 100 / total

            # --- Summary Box ---
            summary_text = (
                f"📌 Spam Count: {spam_count}\n\n"
                f"📌 Ham Count: {ham_count}\n\n"
                f"📌 Total Messages: {total}\n\n"
                f"📌 Spam %: {spam_perc:.2f}%\n\n"
                f"📌 Ham %: {ham_perc:.2f}%"
            )
            st.info(summary_text)

            # --- Filtered DataFrame based on checkboxes + keyword ---
            filtered_df = st.session_state.bulk_df.copy()
            if st.session_state.show_spam and not st.session_state.show_ham:
                filtered_df = filtered_df[filtered_df["result"] == "spam"]
            elif st.session_state.show_ham and not st.session_state.show_spam:
                filtered_df = filtered_df[filtered_df["result"] == "ham"]

            if keyword.strip() != "":
                filtered_df = filtered_df[filtered_df['mgs'].str.contains(keyword, case=False, na=False)]

            # --- Highlight function ---
            def highlight_spam_ham(row):
                if row['result'] == 'spam':
                    return ['background-color: #FFCCCC'] * len(row)  # light red
                elif row['result'] == 'ham':
                    return ['background-color: #CCFFCC'] * len(row)  # light green
                else:
                    return [''] * len(row)

            # --- Display filtered DataFrame with styling ---
            place.dataframe(filtered_df.style.apply(highlight_spam_ham, axis=1), use_container_width=True)

            # --- CSV Export ---
            csv_data = filtered_df.to_csv(index=False).encode("utf-8")
            with r_col2:
                st.download_button(
                    label="📥 Download Results CSV",
                    data=csv_data,
                    file_name="spam_ham_results.csv",
                    mime="text/csv"
                )


#======================SENTIMENT=======================

sw = list(ENGLISH_STOP_WORDS)

    # preserve negations & important sentiment words
preserve_words = ['not', 'no', 'never', 'none', 'nothing', 'nobody', 
                    'but', 'however', 'although', 'though', 
                    'very', 'extremely', 'so', 'too', 'quite', 'slightly', 'barely', 'somewhat', 'always']

for word in preserve_words:
        if word in sw:
            sw.remove(word)
            
def text_cleaning(doc):
        #lowercase
        doc=doc.lower()

        #remove stopwords
        tokens=doc.split()
        new_doc=""
        for t in tokens:
            if t not in sw:
                new_doc=new_doc+" "+t
        new_doc=new_doc.strip()

        #remove chars except alphabets & space
        return re.sub("[^a-z ]","",new_doc)
# Load your trained sentiment model
sentiment_model = joblib.load("models/sentiment.pkl")

def run_sentiment():

    
    st.title("💬 Sentiment Analysis Tool")

    
    

    # ---------------- COLUMNS ----------------
    col1, col2 = st.columns([1, 2], gap='large')

    # ============= LEFT COLUMN (Single Text) =============
    with col1:
        st.subheader("🔹 Single Text Analysis")
        text = st.text_input("Enter text here:", key="single_text")

        if st.button("Analyze Sentiment", key="single_sentiment"):
            if text.strip() != "":
                # --- Show loading ---
                placeholder = st.empty()
                placeholder.info("Analyzing...")

                # --- Predict sentiment ---
                sentiment_label = sentiment_model.predict([text])[0]

                # Optional: if model supports probability
                if hasattr(sentiment_model, "predict_proba"):
                    confidence = sentiment_model.predict_proba([text]).max()
                else:
                    confidence = None

                # --- Clear placeholder ---
                placeholder.empty()

                # --- Fancy display ---
                sentiment_display = {
                    1: ("🔥 Positive", "#CCFFCC"),
                    0: ("💀 Negative", "#FFCCCC")
                }
                text_label, color = sentiment_display.get(sentiment_label, ("❓ Unknown", "#FFFFCC"))

                st.markdown(
                    f"<div style='padding:15px; border-radius:10px; background-color:{color}; font-size:18px; font-weight:bold;'>"
                    f"Sentiment: {text_label}</div>", 
                    unsafe_allow_html=True
                )

                # --- Show confidence bar if available ---
                if confidence:
                    st.progress(confidence)

                # --- Polarity / Subjectivity visual (optional) ---
                from textblob import TextBlob
                tb = TextBlob(text)
                st.text(f"Polarity: {tb.sentiment.polarity:.2f} | Subjectivity: {tb.sentiment.subjectivity:.2f}")

                # --- Copy Result Button ---
                st.button("📋 Copy Sentiment", key=f"copy_{text_label}")

                # --- Optional: Keep history ---
                if 'history' not in st.session_state:
                    st.session_state.history = []
                st.session_state.history.append((text, text_label))
                if len(st.session_state.history) > 5:
                    st.session_state.history.pop(0)

                st.markdown("**Recent Analyses:**")
                for t, s in reversed(st.session_state.history):
                    st.write(f"- {t} ➡️ {s}")


    # ============= RIGHT COLUMN (Bulk Sentiment) =============
    with col2:
        st.subheader("🔹 Bulk Text Analysis")
        
        # File upload
        file = st.file_uploader("Upload CSV/TXT file (one text per row)", type=["csv", "txt"])
        
        if file:
            # Read file once
            if 'bulk_df' not in st.session_state or st.session_state.get('file_changed', True):
                try:
                    st.session_state.bulk_df = pd.read_csv(file, header=None, names=["text"], sep='$#%#')
                except:
                    st.session_state.bulk_df = pd.read_csv(file, header=None, names=["text"], sep='\n')
                st.session_state.predicted = False
                st.session_state.show_pos = True
                st.session_state.show_neg = True
                st.session_state.search_keyword = ''
                st.session_state.file_changed = False
            
            place = st.empty()
            place.dataframe(st.session_state.bulk_df, use_container_width=True)
            
            # --- Predict Bulk Button ---
            r_col1, r_col2 = st.columns([1,1])
            with r_col1:
                if st.button("Predict Bulk", key="bulk_sentiment") and file:
                    placeholder = st.empty()
                    placeholder.info("Analyzing sentiment...")
                    
                    # Predict using your model
                    predictions = sentiment_model.predict(st.session_state.bulk_df["text"])
                    
                    # Convert 0/1 to labels
                    st.session_state.bulk_df['Sentiment'] = ['Positive' if p==1 else 'Negative' for p in predictions]
                    
                    st.session_state.predicted = True
                    placeholder.empty()
            
            # --- Only show filtering if prediction done ---
            if st.session_state.predicted:
                
                # --- Checkboxes for Positive/Negative ---
                c1, c2 = st.columns([1,1])
                with c1:
                    st.session_state.show_pos = st.checkbox("✅ Only Positive", value=st.session_state.show_pos)
                with c2:
                    st.session_state.show_neg = st.checkbox("🛑 Only Negative", value=st.session_state.show_neg)
                
                # --- Keyword search ---
                s_col1, s_col2 = st.columns([4,1])
                with s_col1:
                    st.session_state.search_keyword = st.text_input("🔍 Search Keyword", value=st.session_state.search_keyword)
                with s_col2:
                    st.text('')
                    if st.button("❌ Clear", key="clear_search"):
                        st.session_state.search_keyword = ''
                
                keyword = st.session_state.search_keyword
                
                # --- Filter DataFrame based on checkboxes + keyword ---
                filtered_df = st.session_state.bulk_df.copy()
                if st.session_state.show_pos and not st.session_state.show_neg:
                    filtered_df = filtered_df[filtered_df["Sentiment"] == "Positive"]
                elif st.session_state.show_neg and not st.session_state.show_pos:
                    filtered_df = filtered_df[filtered_df["Sentiment"] == "Negative"]
                
                if keyword.strip() != "":
                    filtered_df = filtered_df[filtered_df['text'].str.contains(keyword, case=False, na=False)]
                
                # --- Highlight positive/negative rows ---
                def highlight_sentiment(row):
                    if row['Sentiment'] == 'Positive':
                        return ['background-color: #CCFFCC'] * len(row)
                    elif row['Sentiment'] == 'Negative':
                        return ['background-color: #FFCCCC'] * len(row)
                    else:
                        return [''] * len(row)
                
                place.dataframe(filtered_df.style.apply(highlight_sentiment, axis=1), use_container_width=True)
                
                # --- CSV Export ---
                csv_data = filtered_df.to_csv(index=False).encode("utf-8")
                with r_col2:
                    st.download_button(
                        label="📥 Download Results CSV",
                        data=csv_data,
                        file_name="sentiment_results.csv",
                        mime="text/csv"
                    )
                
                # --- Summary Section ---
                total_samples = len(filtered_df)
                pos_count = (filtered_df['Sentiment'] == "Positive").sum()
                neg_count = (filtered_df['Sentiment'] == "Negative").sum()
                
                st.markdown("---")
                st.info(f"📊 **Summary:**\n- Total Samples: {total_samples}\n- Positive: {pos_count}\n- Negative: {neg_count}")




# Load model inside function
def load_news_model():
    return joblib.load("models/news.pkl")


# -------------------------------------------------------------------
# 📌 NEWS DETECTION MAIN FUNCTION
# -------------------------------------------------------------------
def run_news_detection():

    st.title("📰 News Category Detection")
    st.write("Classify text/news into **World, Sports, Business, Sci/Tech**")

    # Load model
    news_model = load_news_model()

    # Category mapping (optional)
    label_map = {
        "World": "World",
        "Sports": "Sports",
        "Business": "Business",
        "Sci/Tech": "Sci/Tech"
    }

    # ---------------- COLUMNS ----------------
    col1, col2 = st.columns([1, 2], gap='large')

    # ===================================================================
    # 📌 LEFT COLUMN — Single News Detection
    # ===================================================================
    with col1:
        st.subheader("🔹 Single Text Classification")

        # Custom CSS for glow/hover effect
        st.markdown("""
            <style>
            .category-box {
                padding: 18px; 
                border-radius: 12px; 
                font-size: 18px; 
                font-weight: bold; 
                transition: 0.3s;
                border: 2px solid #eee;
            }
            .category-box:hover {
                transform: scale(1.02);
                box-shadow: 0 0 15px rgba(0,0,0,0.15);
            }
            .mini-card {
                padding:10px; 
                border-radius:10px;
                background:#ffffffdd;
                margin-top:10px;
                border:1px solid #eee;
            }
            </style>
        """, unsafe_allow_html=True)

        text = st.text_input("Enter news text:", key="single_news")

        if st.button("🎯 Predict Category", key="single_news_button"):
            if text.strip() != "":
                placeholder = st.empty()
                placeholder.info("🔍 Classifying... Please wait")

                # ----- Prediction -----
                pred = news_model.predict([text])[0]

                # Probability (optional)
                if hasattr(news_model, "predict_proba"):
                    proba = news_model.predict_proba([text])[0]
                    confidence = max(proba)
                else:
                    proba, confidence = None, None

                placeholder.empty()

                category = label_map.get(pred, "Unknown")

                # ----- Colors -----
                colors = {
                    "World": "#D6EAF8",
                    "Sports": "#D4EFDF",
                    "Business": "#FCF3CF",
                    "Sci/Tech": "#F5EEF8",
                    "Unknown": "#F2F4F4"
                }

                emojis = {
                    "World": "🌍",
                    "Sports": "🏆",
                    "Business": "💼",
                    "Sci/Tech": "🧪",
                    "Unknown": "❓"
                }

                # ----- Category Display -----
                st.markdown(
                    f"""
                    <div class="category-box" style="background:{colors[category]};">
                        Category: {emojis[category]} <b>{category}</b>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # ----- Confidence Gauge -----
                if confidence:
                    st.write(f"**Model Confidence:** `{confidence*100:.2f}%`")
                    st.progress(confidence)

                # ----- Probability Breakdown -----
                if proba is not None:
                    st.markdown("### 🔬 Probability Breakdown")
                    pb_df = pd.DataFrame({
                        "Category": list(label_map.values()),
                        "Probability": proba
                    }).sort_values("Probability", ascending=False)

                    st.dataframe(pb_df, hide_index=True, use_container_width=True)

                # ----- Mini Summary Card -----
                st.markdown(
                    f"""
                    <div class="mini-card">
                        <b>📝 Summary</b><br>
                        • Input length: {len(text.split())} words<br>
                        • Top category: {category}<br>
                        • {("Confidence: " + str(round(confidence*100,2)) + "%") if confidence else ""}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # ----- Copy Button -----
                st.button("📋 Copy Result", key=f"copy_{category}")

    # ===================================================================
    # 📌 RIGHT COLUMN — BULK NEWS FILE
    # ===================================================================
    with col2:
        st.subheader("🔹 Bulk News Analysis")

        file = st.file_uploader("Upload CSV/TXT file", type=["csv", "txt"])

        if file:

            # Read file only first time
            if 'news_bulk_df' not in st.session_state or st.session_state.get('news_file_changed', True):
                try:
                    st.session_state.news_bulk_df = pd.read_csv(file, header=None, names=["text"], sep='$#%#')
                except:
                    st.session_state.news_bulk_df = pd.read_csv(file, header=None, names=["text"], sep='\n')

                st.session_state.news_predicted = False
                st.session_state.show_filters = [True] * 4
                st.session_state.news_search_keyword = ""
                st.session_state.news_file_changed = False

            place = st.empty()
            place.dataframe(st.session_state.news_bulk_df, use_container_width=True)

            # Predict Button
            r1, r2 = st.columns([1,1])
            with r1:
                if st.button("Predict Bulk", key="predict_bulk_news"):

                    placeholder = st.empty()
                    placeholder.info("Classifying all rows...")

                    # Predict
                    preds = news_model.predict(st.session_state.news_bulk_df["text"])

                    st.session_state.news_bulk_df["Category"] = [label_map[p] for p in preds]

                    st.session_state.news_predicted = True
                    placeholder.empty()

            # Filtering UI only after prediction
            if st.session_state.news_predicted:

                # Category Filters
                categories = ["World", "Sports", "Business", "Sci/Tech"]

                c1, c2, c3, c4 = st.columns(4)
                st.session_state.show_filters = [
                    c1.checkbox("🌍 World", value=st.session_state.show_filters[0]),
                    c2.checkbox("🏆 Sports", value=st.session_state.show_filters[1]),
                    c3.checkbox("💼 Business", value=st.session_state.show_filters[2]),
                    c4.checkbox("🔬 Sci/Tech", value=st.session_state.show_filters[3])
                ]

                # Keyword Search
                s1, s2 = st.columns([4,1])
                with s1:
                    st.session_state.news_search_keyword = st.text_input(
                        "🔍 Search Keyword",
                        value=st.session_state.news_search_keyword
                    )

                with s2:
                    st.text("")
                    if st.button("❌ Clear", key="clear_news_search"):
                        st.session_state.news_search_keyword = ""

                keyword = st.session_state.news_search_keyword

                # Filtering logic
                df = st.session_state.news_bulk_df.copy()

                allowed_categories = [cat for cat, allow in zip(categories, st.session_state.show_filters) if allow]
                df = df[df["Category"].isin(allowed_categories)]

                if keyword.strip():
                    df = df[df["text"].str.contains(keyword, case=False, na=False)]

                # Coloring rows
                def highlight_row(row):
                    col_map = {
                        "World": "background-color: #D6EAF8",
                        "Sports": "background-color: #D4EFDF",
                        "Business": "background-color: #FCF3CF",
                        "Sci/Tech": "background-color: #F5EEF8"
                    }
                    return [col_map.get(row['Category'], "")] * len(row)

                place.dataframe(
                    df.style.apply(highlight_row, axis=1),
                    use_container_width=True
                )

                # CSV Export
                csv_data = df.to_csv(index=False).encode("utf-8")
                with r2:
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv_data,
                        file_name="news_results.csv",
                        mime="text/csv"
                    )

                # ------ SMART SUMMARY CARDS ------
                st.markdown("### 📊 Summary Insights")
                st.markdown("""
                    <div class="summary-card">
                        <b>Total Rows:</b> """ + str(len(df)) + """<br>
                        <b>World:</b> """ + str((df['Category']=='World').sum()) + """ |
                        <b>Sports:</b> """ + str((df['Category']=='Sports').sum()) + """ |
                        <b>Business:</b> """ + str((df['Category']=='Business').sum()) + """ |
                        <b>Sci/Tech:</b> """ + str((df['Category']=='Sci/Tech').sum()) + """
                    </div>
                """, unsafe_allow_html=True)
#=================================================
#============DIABETES PREDICTION (PRO)============
#=================================================

diabetes_model = joblib.load("models/diabetes.pkl")

def run_Diabetes_Prediction():

    coll1, coll2 = st.columns([3, 1],gap='small')

    with coll1:
        st.markdown("## 🩺 Diabetes Risk Prediction (AI-Powered)")

    with coll2:
        st.markdown("""
        <div style='font-size:14px; color:gray; padding-top:12px; text-align:right;'>
        ⚠️ <b>Disclaimer:</b> This prediction is based on statistical AI modeling.<br>
        It is <u>not</u> a medical diagnosis. Please consult a healthcare professional.
        </div>
        """, unsafe_allow_html=True)


    # Helper for Y/N
    def yn_to_int(x):
        if x is None or x.strip() == "":
            return None
        return 1 if x.lower() == "y" else 0

    st.markdown("### ✏️ Fill Your Health Information")

    col1, col2 = st.columns(2)

    with col1:
        HighBP = st.text_input("🩸 High Blood Pressure (y/n)")
        HighChol = st.text_input("🧬 High Cholesterol (y/n)")
        CholCheck = st.text_input("🧪 Cholesterol Check 5yrs (y/n)")
        Smoker = st.text_input("🚬 Smoker? (y/n)")
        Stroke = st.text_input("⚡ Stroke history? (y/n)")
        HeartDisease = st.text_input("❤️ Heart disease/attack? (y/n)")
        PhysActivity = st.text_input("🏃 Physical Activity? (y/n)")
        Fruits = st.text_input("🍎 Eat fruits daily? (y/n)")

    with col2:
        Veggies = st.text_input("🥦 Eat vegetables daily? (y/n)")
        HvyAlcohol = st.text_input("🍺 Heavy alcohol consumption? (y/n)")
        AnyHealthcare = st.text_input("🏥 Healthcare coverage? (y/n)")
        Sex = st.text_input("⚧ Gender (m/f)")
        Age = st.text_input("📅 Age group (0–13)")
        BMI = st.number_input("⚖️ BMI Value", min_value=10.0, value=25.0)

        st.markdown("##### Don't know BMI? Enter height & weight 👇")
        hcol1, hcol2 = st.columns(2)

        with hcol1:
            height_cm = st.number_input("Height (cm)", min_value=100.0, value=170.0)

        with hcol2:
            weight_kg = st.number_input("Weight (kg)", min_value=30.0, value=65.0)

        # Auto BMI update
        auto_bmi = weight_kg / ((height_cm / 100) ** 2)

        st.info(f"📌 Your BMI 😏: **{auto_bmi:.2f}**")

        # If user didn't change BMI, use auto BMI
        if BMI == 25.0:
            BMI = auto_bmi


    # Feature vector builder
    def get_features():
        try:
            f = [
                yn_to_int(HighBP),
                yn_to_int(HighChol),
                yn_to_int(CholCheck),
                float(BMI),
                yn_to_int(Smoker),
                yn_to_int(Stroke),
                yn_to_int(HeartDisease),
                yn_to_int(PhysActivity),
                yn_to_int(Fruits),
                yn_to_int(Veggies),
                yn_to_int(HvyAlcohol),
                yn_to_int(AnyHealthcare),
                1 if Sex.lower() == "m" else 0,
                int(Age)
            ]
            if None in f:
                return None
            return f
        except:
            return None


    # Predict Button
    if st.button("🔍 Predict Diabetes Risk"):

        features = get_features()
        if features is None:
            st.error("⚠️ Please fill all fields correctly (only y/n and numbers).")
            return

        with st.spinner("Analyzing your health profile... ⏳"):
            x = np.array(features).reshape(1, -1)
            pred = diabetes_model.predict(x)[0]

            if hasattr(diabetes_model, "predict_proba"):
                confidence = diabetes_model.predict_proba(x).max()
            else:
                confidence = None

        # RESULT DISPLAY BOX
        label_map = {
            0: ("🟢 No Diabetes", "#D4EFDF"),
            1: ("🟡 Pre-Diabetes", "#FCF3CF"),
            2: ("🔴 Diabetes", "#F5B7B1")
        }

        result_text, color = label_map.get(pred, ("❓ Unknown", "#F8F9F9"))

        st.markdown(
            f"""
            <div style="
                padding:20px;
                border-radius:15px;
                background: {color};
                text-align:center;
                font-size:22px;
                font-weight:bold;
                box-shadow: 0px 0px 10px #ccc;
            ">
                Prediction: {result_text}
            </div>
            """,
            unsafe_allow_html=True
        )

        # ============================
        #  NEW FEATURE 1 — RISK METER
        # ============================
        st.markdown("### 🎯 Risk Meter")
        st.progress(confidence if confidence else 0.5)

        # ============================
        # NEW FEATURE 2 — SUMMARY CARD
        # ============================
        st.markdown("### 🧠 Smart Health Decision")

        if pred == 0:
            st.success("👍 Your profile shows **no diabetes risk**. Keep living healthy!")
        elif pred == 1:
            st.warning("🟡 **Pre-diabetes indicators found.** Improve diet & daily routine soon.")
        else:
            st.error("🔴 **High diabetes risk detected!** Consult a doctor for medical tests.")

        # Extra Cool Stats Cards
        st.markdown("### 📊 Your Summary")
        c1, c2, c3 = st.columns(3)

        c1.metric("⚖️ BMI", f"{BMI}")
        c2.metric("📅 Age Group", f"{Age}")
        c3.metric("🔥 Risk Score", f"{round(confidence*100,1)}%" if confidence else "N/A")

        # ============================
        # NEW FEATURE 4 — LIFESTYLE TIPS
        # ============================
        st.markdown("### 🍀 Personalized Lifestyle Advice")

        tips = []

        if BMI > 28:
            tips.append("⚖️ *Your BMI is high — daily 30 min walk recommended.*")
        if yn_to_int(Smoker) == 1:
            tips.append("🚭 *Quitting smoking drastically reduces diabetes risk.*")
        if yn_to_int(PhysActivity) == 0:
            tips.append("🏃 *Start light exercise 3–4 times a week.*")
        if yn_to_int(Fruits) == 0 or yn_to_int(Veggies) == 0:
            tips.append("🥗 *Add more fruits & vegetables to your meals.*")
        if yn_to_int(HvyAlcohol) == 1:
            tips.append("🍺 *Reduce alcohol consumption.*")

        if len(tips) == 0:
            st.info("🎉 You already maintain a very healthy lifestyle!")
        else:
            for t in tips:
                st.write(t)

        # ============================
        # NEW FEATURE 5 — RISK FACTOR BADGES
        # ============================
        st.markdown("### 🏷️ Risk Factor Badges")
        risk_cols = st.columns(4)

        badges = [
            ("High BP", HighBP),
            ("Cholesterol", HighChol),
            ("Smoker", Smoker),
            ("Heart Disease", HeartDisease),
            ("Stroke", Stroke),
            ("Alcohol", HvyAlcohol)
        ]

        idx = 0
        for name, val in badges:
            col = risk_cols[idx % 4]
            if yn_to_int(val) == 1:
                col.error(f"❗ {name}")
            else:
                col.success(f"✅ {name}")
            idx += 1

#================== CAR PRICE======================================
class SimpleCarModel:
    def __init__(self):
        # Encoders + scaler + model
        self.le_brand = LabelEncoder()
        self.le_model = LabelEncoder()
        self.scaler = None
        self.model = None

        # Mappings for categorical columns
        self.trans_map = {"Manual":1, "Automatic":2}
        self.fuel_map = {"Petrol":1, "Diesel":2, "hybrid":3, "Hybrid/CNG":4}
        self.owner_map = {"first":1, "second":2}

    # -------------------------
    # TRAIN
    # -------------------------
    def fit(self, df):
        df = df.copy()

        # Encode Brand + model
        df['Brand'] = self.le_brand.fit_transform(df['Brand'])
        df['model'] = self.le_model.fit_transform(df['model'])

        # Encode already numeric columns
        df['Transmission'] = df['Transmission'].map(self.trans_map).astype(int)
        df['FuelType'] = df['FuelType'].map(self.fuel_map).astype(int)
        df['Owner'] = df['Owner'].map(self.owner_map).astype(int)

        # Prepare X, y
        X = df.drop("AskPrice", axis=1).values
        y = df["AskPrice"].values

        # Scale
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Train model
        self.model = DecisionTreeRegressor()
        self.model.fit(X_scaled, y)

        print("Model trained successfully!")

    # -------------------------
    # PREDICT
    # -------------------------
    def predict(self, sample):
        
        X_new = pd.DataFrame([sample])

        # Encode categorical
        X_new['Brand'] = self.le_brand.transform([X_new['Brand'][0]])
        X_new['model'] = self.le_model.transform([X_new['model'][0]])
        X_new['Transmission'] = self.trans_map[X_new['Transmission'][0]]
        X_new['FuelType'] = self.fuel_map[X_new['FuelType'][0]]
        X_new['Owner'] = self.owner_map[X_new['Owner'][0]]

        # Convert to array and scale
        X_scaled = self.scaler.transform(X_new.values)

        # Predict
        return self.model.predict(X_scaled)[0]

# ----------------------------
# Load trained model
# ----------------------------
model = joblib.load("models/car1.pkl")

def car_price():

    # ----------------------------
    # Options for drop-downs
    # ----------------------------
    brands = [
        "Maruti Suzuki", "Hyundai", "Honda", "Toyota", "Mahindra", "Mercedes-Benz",
        "Tata", "Volkswagen", "BMW", "Ford", "Renault", "Audi", "Skoda", "Kia",
        "Chevrolet", "Nissan", "MG", "Jeep", "Land Rover", "Volvo", "Jaguar",
        "Datsun", "Mini", "Porsche", "Fiat", "Lexus", "Mitsubishi", "Isuzu",
        "Force", "Ssangyong", "Ambassador", "Toyota Land", "Bajaj", "Citroen",
        "Rolls-Royce", "Opel", "Aston Martin", "ICML", "Ashok", "Maserati",
        "Bentley", "Lamborghini", "Hummer"
    ]

    models_list = ['City', 'Innova', 'VentoTest', 'Swift', 'Baleno', 'X3', '5 Series', 'maruti-suzuki-dzire', 'Ecosport', 'Alto-K10', 'Carnival', 'Swift-Dzire', 'Corolla', 'GLE COUPE', 'Xcent', 'Seltos', 'Ertiga', '3 Series GT', 'Endeavour', 'Innova Crysta', 'A3', 'KWID', 'Hector', 'Celerio', 'Vitara-Brezza', '2.8 Legender 4X2', 'S90', 'Venue', 'Creta', 'Alcazar', 'i20', 'E-Class', 'Polo', 'Verna', 'A4', 'Fortuner', 'C-Class', 'Kushaq', 'Ciaz', 'Safari', 'BRV', 'Duster', 'Wagon-R', 'Bolero Power Plus', 'Eon', 'Hector Plus', 'XUV500', 'GLS', 'i10', 'GLA Class', 'Carens', 'Ignis', 'Grand i10', 'Getz Prime', 'Ritz', 'Sonet', 'GLC Coupe', 'Scorpio-N', 'Cooper 3 DOOR',
                    'Nexon', 'Etios', 'CrossPolo', 'Vento', 'Range Rover Evoque', 'Indigo Ecs', 'Thar', 'Camry', 'Bolero', 'Brio', 'City ZX', 'WRV', 'Discovery Sport', 'Harrier', 'Zest', 'Eeco', 'Swift Dzire', 'Sx4', 'Kodiaq', 'Altroz', 'Grand i10 Nios', 'Alto-800', 'X1', 'X5', 'Rapid', 'TUV', 'Alto K10', 'M-Class', 'Glanza', 'Compass', 'Omni', 'Cruze', 'Amaze', 'Jazz', 'Scala', 'S-Presso', 'Tucson', 'Alto', 'Santa Fe', 'Dzire', 'ASTOR', 'Q7', 'Cooper 5 DOOR', 'Q3', 'Hexa', 'Creta Facelift', '7 Series', 'Bolt', 'XF', 'Tiago', 'Indigo Marina', 'Nano Genx', 'City Hybrid eHEV', 'Tiguan',
                    'G', '3 Series', 'Range Rover', 'Grand Vitara', 'XC 90', 'A6', 'Polo GTI', 'Virtus', 'A-Star', 'Cooper S', 'Etios Liva', 'CLA', 'XUV700', 'Alturas G4', 'Scorpio', 'New i20', 'Alto 800', 'Elite i20', 'Defender', 'Santro Xing', 'Linea', 'Vanquish', 'maruti-suzuki-brezza', 'A8 L', 'V40 Cross Country', 'Captur', 'Passat', 'KUV 100', 'Bolero Neo', 'Elantra', 'Macan', 'Terrano', 'Swift-Dzire-Tour', 'XL6', 'Xylo', 'S-Class', '6 Series GT', 'Figo', 'Ameo', 'Innova Hycross', 'Triber', 'Q5', 'Motors FM Gurkha', 'Jeep', '718', 'Punto', 'Corolla Altis', 'Pajero Sport', 'Manza', 'Cooper Convertible', 'Scorpio Classic', 'CRV', 'ES', 'GL-Class',
                      'Tigor', 'X7', 'Octavia', 'Wagon R', 'S-Cross', 'Gypsy', 'Renault Logan', 'Boxster', 'Indigo Cs', 'XC40', 'Lodgy', 'Nano', 'XC60', 'Z4', 'Civic', 'Urban Cruiser', 'Brezza', 'Punch', 'Cedia', 'GLC', 'Sonata Embera', 'Figo Aspire', 'Accord', '3 DOOR', 'MAX', 'G Class', 'Indica Vista', 'RediGO', 'CLS-Class', 'VELLFIRE', 'Superb', 'Zen-Estilo', 'Cooper', 'Estilo', 'X4', 'LX', 'Enjoy', 'Mobilio', 'Punto EVO', 'Bolero Neo Plus', 'Scorpio N', 'D-Max V-Cross', 'Fluidic Verna', 'G-Class', 'XUV 300', 'Optra', ' Wagon R', 'Cooper Countryman', 'GLE', 'GLC Class', 'Fiesta Classic', 'MUX', 'V-Class',
                      'Safari Storme', 'Indica Ev2', 'Celerio-X', 'Land Cruiser', 'Indica', 'MICRA PRIMO', 'Tavera', 'S-Cross1', 'S60 Cross Country', 'Slavia', 'Punch ', '1000', 'XJ', 'Fronx', 'Beat', 'Santro', 'S60', 'Cayenne', 'Range Rover Sport', 'Verito', 'WR-V', 'Laura', '800', 'Range Rover Velar', 'Jetta', 'Wagon-R-1-0', 'Freelander 2', 'Spark', 'Discovery', 'C Class', 'Micra', 'Kiger', 'Ambassador', 'Marazzo', 'i20 Active', 'Redi Go', '2 Series', 'Esteem', 'Outlander', 'Etios Cross', 'GLA', 'GLE Class', 'Accent', 'Marshal', 'Fabia', 'A5', 'Jimny', 'AMG GLE Coupe', 'Countryman', 'Sail U-VA', 'Fluence', 'NX',
                        'Fiesta', 'Pulse', 'Sonata', 'Tiago Nrg', '1 Series', 'S 80', 'Aura', '6 Series', 'Meridian', 'Kicks', 'Vitara Brezza', '3 Series Gran Limousine', 'PATROL', 'B Class', 'TUV 300', 'Sail', 'Yaris', 'Sumo Grande MK II', 'S80', 'Celerio X', 'EON', 'Motors FM Trax Cruiser', 'XUV300', 'Sunny', 'Sumo Gold', 'OptraSRV', 'Indica V2', 'X6', 'Magnite', 'KUV100 NXT', 'Gloster', 'Urban Cruiser Hyryder', 'Trailblazer', 'Quanto', 'Sumo Victa', 'i20 N Line', 'Xenon XT', 'Lancer', 'Land Cruiser Prado', 'Taigun', 'Zen Estilo', 'Cooper Clubman', 'Beetle', 'A-Class Limousine', 'XE', 'Venue N Line', 'CR-V', 'tata-punch', 'Cayenne Coupe', 'F-Pace',
                        'Grand Punto', 'GO', 'Phantom Series II', '5 Series Gt', 'Venture', 'Aspire', 'LS', 'Mustang', 'Tiguan All Space', 'Free Style', 'H5x', 'XC90', 'TUV 300-plus', 'New Santro', 'Wagon R 1.0',
                    'AMG C 43', 'New Elantra', 'Motors FM Force One Test', 'New-gen Swift', 'Punto Pure', 'Rhino Rx', 'New Accord', 'RX', 'Xcent Prime', 'E-20', 'RE60', 'New Duster', 'Ssangyong-Rexton', 'Grand i10 Prime',
                      'Wrangler', 'SLK-Class', 'SX4', 'Phantom Drop HeadCoupe', 'Nuvosport', 'Panamera', 'Wagon-R-Stingray', '5 DOOR', 'Micra Active', 'Cayman', 'AMG A35', 'Optra Magnum', 'M340i', 'GTI', 'Getz', 'X5 M', 'Baleno-RS',
                        'A Class', 'Exter', 'Invicto', 'Rexton', 'Sumo', 'Stingray', 'Teana', 'Corsa', 'Fronx ', 'Accent Hatchback', 'SL-Class', 'Vista Tech', 'Captiva', 'GO Plus', 'A3 Cabriolet', 'Leyland Stile', 'Flying Spur', 'Curvv',
                          'Estima', 'Ikon', 'Avventura', 'Others', 'Ssangyong Rexton', 'A-class Limousine', '2.9 Sportback', 'CERATO', 'Elevate', 'Urvan', 'Hi-Lande Isuzu Hi-Lander', 'AMG', 'Urban Cross', 'Gran Turismo', 'M2', 'Verito Vibe', 'Qualis', 'Aria', 'New Verna', 'Cruiser', 'Grande Punto', 'MU-X', 'Karoq', 'Q3 Sportback', 'Logan', 'Thar Roxx', 'Escort', 'Maxx', 'Five-door Thar', 'C5 Aircross', 'Cruiser Prado', 'Pajero', 'S Cross', 'RS7', 'Punch EV', 'Urus', 'XUV 3XO', 'Harrier EV', '718 CAYMAN', 'Q2', 'H2', 'R-Class', 'Imperio', 'Sonata Transform', 'A8', 'Versa', 'T-ROC', 'X-Trail', 'Ventury', 'Yeti']


    trans_options = ["Automatic", "Manual"]
    fuel_options = ["Petrol", "Diesel", "hybrid", "Hybrid/CNG"]
    owner_options = ["first", "second"]

    # ----------------------------
    # Streamlit GUI
    # ----------------------------
    colT1, colT2 = st.columns([2, 1])

    with colT1:
        st.markdown("## 🚗 Car Resale Value Prediction")

    with colT2:
        st.markdown(
            """
                        <div style='font-size:14px; color:gray; padding-top:12px;'>
            ⚠️ *Disclaimer:* This price prediction is entirely based on statistical modeling.  
            Actual market value may vary depending on the car's condition, location, demand, and negotiation.
            </div>

            """,
            unsafe_allow_html=True
        )


    col1, col2 = st.columns(2)

    with col1:
        Brand = st.selectbox("Brand", ["-- Select Brand --"] + brands)
        model_name = st.selectbox("Model", ["-- Select Model --"] + models_list)
        Year = st.number_input("Year", min_value=1990, max_value=2025, value=2017)
        kmDriven = st.number_input("Kilometers Driven", min_value=0, value=50000)

    with col2:
        Transmission = st.selectbox("Transmission", ["-- Select --"] + trans_options)
        Owner = st.selectbox("Owner", ["-- Select --"] + owner_options)
        FuelType = st.selectbox("Fuel Type", ["-- Select --"] + fuel_options)
    st.markdown("### 📝 Additional Car Details (Optional)")
    extra_details = st.text_area(
    "Add any special notes about your car:",
    placeholder="Example: New tyres installed, recently serviced, no accidental history, single owner...",
    height=120
    )

    # ----------------------------
    # Predict Button
    # ----------------------------
    
    if st.button("Predict Price"):
        if Brand.startswith("--") or model_name.startswith("--") or Transmission.startswith("--") or Owner.startswith("--") or FuelType.startswith("--"):
            st.warning("⚠️ Please select all dropdown fields before predicting!")
        else:
            test_sample = {
                "Brand": Brand,
                "model": model_name,
                "Year": Year,
                "kmDriven": kmDriven,
                "Transmission": Transmission,
                "Owner": Owner,
                "FuelType": FuelType
            }
            
            price = model.predict(test_sample)
            #st.success(f"💰 Predicted Price: ₹ {price:.2f}")
        

            # -----------------------------
            # Premium Price Output
            # -----------------------------
            st.markdown("### 💰 **Predicted Sell Price**")
            st.success(f"₹ {price:.2f}")

            # Price Range
            lower = price * 0.90
            upper = price * 1.10
            st.markdown(f"""**Estimated Market Range:** 
                        ₹ {lower:.2f} – ₹ {upper:.2f}""")

            # -----------------------------
            # Metric Cards
            # -----------------------------
            c1, c2 = st.columns(2)
            c1.metric("Model Year", Year)
            c2.metric("KM Driven", f"{kmDriven:,}")

            # -----------------------------
            # Price Meter
            # -----------------------------
            st.markdown("### 📊 Price Positioning")
            max_price = 2000000
            progress_val = min(price / max_price, 1.0)
            st.progress(progress_val)
            
            
#=================================================
#========= STUDENT PERFORMANCE PREDICTION (PRO) ===
#=================================================

student_model = joblib.load("models/student.pkl")

def run_Student_Performance():

    coll1, coll2 = st.columns([3, 1], gap='small')

    with coll1:
        st.markdown("## 🎓 Student Performance Prediction (AI-Based)")

    with coll2:
        st.markdown("""
        <div style='font-size:14px; color:gray; padding-top:12px; text-align:right;'>
        ⚠️ <b>Note:</b> This prediction is based on AI statistical modeling.<br>
        It is <u>not</u> an academic or psychological assessment.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### ✏️ Enter Student Details")

    # ==============================
    # Encoding helper functions
    # ==============================
    def map_parental_involvement(x):
        return {"Low": 1, "Medium": 2, "High": 3}.get(x)

    def map_access_resources(x):
        return {"Low": 1, "Medium": 2, "High": 3}.get(x)

    def map_extracurricular(x):
        return {"No": 1, "Yes": 2}.get(x)

    def map_motivation(x):
        return {"Low": 1, "Medium": 2, "High": 3}.get(x)

    def map_internet(x):
        return {"No": 1, "Yes": 2}.get(x)

    def map_family_income(x):
        return {"Low": 1, "Medium": 2, "High": 3}.get(x)

    def map_teacher_quality(x):
        return {"Low": 1, "Medium": 2, "High": 3}.get(x)

    def map_school_type(x):
        return {"Public": 1, "Private": 2}.get(x)

    def map_peer(x):
        return {"Negative": 1, "Neutral": 2, "Positive": 3}.get(x)

    def map_learning(x):
        return {"No": 1, "Yes": 2}.get(x)

    def map_education(x):
        return {"High School": 1, "College": 2, "Postgraduate": 3}.get(x)

    def map_distance(x):
        return {"Near": 1, "Moderate": 2, "Far": 3}.get(x)

    def map_gender(x):
        return {"Male": 1, "Female": 2}.get(x)

    # ==============================
    # FORM UI
    # ==============================
    col1, col2 = st.columns(2)

    with col1:
        Hours_Studied = st.number_input("📘 Hours Studied", min_value=0, value=0)
        Attendance = st.number_input("🏫 Attendance (%)", min_value=0, max_value=100, value=0)
        Parental_Involvement = st.selectbox("👨‍👩‍👦 Parental Involvement", ["--- Select ---","Low", "Medium", "High"])
        Access_to_Resources = st.selectbox("📚 Access to Resources", ["--- Select ---","Low", "Medium", "High"])
        Extracurricular_Activities = st.selectbox("🎨 Extracurricular Activities", ["--- Select ---","No", "Yes"])
        Sleep_Hours = st.number_input("😴 Sleep Hours / day", min_value=0, value=0)
        Previous_Scores = st.number_input("📝 Previous Score (%)", min_value=0, max_value=100, value=0)
        Motivation_Level = st.selectbox("🔥 Motivation Level", ["--- Select ---","Low", "Medium", "High"])
        Internet_Access = st.selectbox("🌐 Internet Access", ["No", "Yes"])

    with col2:
        Tutoring_Sessions = st.number_input("📖 Tutoring Sessions per week", min_value=0, value=0)
        Family_Income = st.selectbox("💰 Family Income", ["--- Select ---","Low", "Medium", "High"])
        Teacher_Quality = st.selectbox("👩‍🏫 Teacher Quality", ["--- Select ---","Low", "Medium", "High"])
        School_Type = st.selectbox("🏫 School Type", ["--- Select ---","Public", "Private"])
        Peer_Influence = st.selectbox("🧑‍🤝‍🧑 Peer Influence", ["--- Select ---","Negative", "Neutral", "Positive"])
        Physical_Activity = st.number_input("⚽ Physical Activity (hrs/week)", min_value=0, value=0)
        Learning_Disabilities = st.selectbox("🧠 Learning Disabilities", ["--- Select ---","No", "Yes"])
        Parental_Education_Level = st.selectbox("🎓 Parent Education Level", ["--- Select ---","High School", "College", "Postgraduate"])
        Distance_from_Home = st.selectbox("📍 Distance from Home", ["--- Select ---","Near", "Moderate", "Far"])
        Gender = st.selectbox("⚧ Gender", ["--- Select ---","Male", "Female"])

    # ==============================
    # Feature Vector
    # ==============================
    def build_features():
        try:
            return [
                Hours_Studied,
                Attendance,
                map_parental_involvement(Parental_Involvement),
                map_access_resources(Access_to_Resources),
                map_extracurricular(Extracurricular_Activities),
                Sleep_Hours,
                Previous_Scores,
                map_motivation(Motivation_Level),
                map_internet(Internet_Access),
                Tutoring_Sessions,
                map_family_income(Family_Income),
                map_teacher_quality(Teacher_Quality),
                map_school_type(School_Type),
                map_peer(Peer_Influence),
                Physical_Activity,
                map_learning(Learning_Disabilities),
                map_education(Parental_Education_Level),
                map_distance(Distance_from_Home),
                map_gender(Gender)
            ]
        except:
            return None

    # ==============================
    # Predict Button
    # ==============================
    if st.button("🔍 Predict Performance"):

        feats = build_features()
        if None in feats:
            st.error("⚠️ Please fill all details properly.")
            return

        with st.spinner("Analyzing student performance... ⏳"):
            x = np.array(feats).reshape(1, -1)
            pred = student_model.predict(x)[0]

        # ==============================
        # Result Box
        # ==============================
        if pred >= 85:
            label = ("🟢 Excellent Performance", "#D5F5E3")
        elif pred >= 60:
            label = ("🟡 Average Performance", "#FCF3CF")
        else:
            label = ("🔴 Poor Performance", "#F5B7B1")

        text, color = label

        st.markdown(
            f"""
            <div style="
                padding:20px;
                border-radius:15px;
                background:{color};
                font-size:24px;
                font-weight:bold;
                text-align:center;">
                {text} <br>
                🎯 Predicted Score: {pred:.2f}%
            </div>
            """,
            unsafe_allow_html=True
        )

        # ==============================
        # Summary Cards
        # ==============================
        st.markdown("### 📊 Student Summary")
        c1, c2, c3 = st.columns(3)
        c1.metric("Hours Studied", Hours_Studied)
        c2.metric("Previous Score", Previous_Scores)
        c3.metric("Motivation", Motivation_Level)

        # ==============================
        # Smart Suggestions
        # ==============================
        st.markdown("### 💡 Personalized Tips")

        tips = []

        if Hours_Studied < 2:
            tips.append("📘 Increase study time to at least 2–3 hrs/day.")
        if Sleep_Hours < 7:
            tips.append("😴 Improve sleep cycle (7–8 hrs ideal).")
        if Motivation_Level == "Low":
            tips.append("🔥 Try small goals to boost motivation.")
        if map_extracurricular(Extracurricular_Activities) == 1:
            tips.append("🎨 Join at least 1 extracurricular activity.")
        if map_internet(Internet_Access) == 1:
            tips.append("🌐 Internet access important for modern learning.")

        if len(tips) == 0:
            st.success("🎉 Student profile looks healthy and balanced!")
        else:
            for t in tips:
                st.write(t)

 

module = st.session_state["module"]
task = st.session_state["task"]

if module == "Classification":
    if task == "Spam Detection":
        run_spam()
    elif task == "News Detection":
        run_news_detection()
    elif task == "Diabetes Prediction":
        run_Diabetes_Prediction()

elif module == "Linear Regression Models":
    if task == "Car Price Prediction":
        car_price()
    elif task == "Student Performance Prediction":
        run_Student_Performance()

elif module =='Home':
    home()

elif module=='Sentiment Analysis':
    run_sentiment()
elif module == "User Login":
    user_login()     # ← tumhara function

elif module == "Create User":
    create_user()

elif module == "AdminLogin":
    admin_login_ui()

elif module == "AdminDashboard":
    admin_dashboard_ui()

elif module == "ABOUT Me":
    about_me_ui()
