import streamlit as st
import numpy as np
import pandas as pd
import time
import random
import joblib
import os
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.models import load_model
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import plotly.graph_objects as go

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Burnout Risk Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Game Containers
st.markdown("""
    <style>
    .game-container {
        height: 400px;
        display: flex;
        align-items: center;
        justify-content: center;
        background-color: #1e3a8a;
        border-radius: 10px;
        margin: 20px 0;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================
if 'current_page' not in st.session_state:
    st.session_state.current_page = 0

# PVT State
if 'pvt_rts' not in st.session_state:
    st.session_state.pvt_rts = []
if 'pvt_completed' not in st.session_state:
    st.session_state.pvt_completed = False
if 'pvt_trial' not in st.session_state:
    st.session_state.pvt_trial = 0
if 'pvt_show_stimulus' not in st.session_state:
    st.session_state.pvt_show_stimulus = False
if 'pvt_can_click' not in st.session_state:
    st.session_state.pvt_can_click = False
if 'pvt_start_time' not in st.session_state:
    st.session_state.pvt_start_time = 0

# SART State
if 'sart_completed' not in st.session_state:
    st.session_state.sart_completed = False
if 'sart_trial' not in st.session_state:
    st.session_state.sart_trial = 0
if 'sart_current_number' not in st.session_state:
    st.session_state.sart_current_number = None
if 'sart_can_respond' not in st.session_state:
    st.session_state.sart_can_respond = False
if 'sart_responses' not in st.session_state:
    st.session_state.sart_responses = []
if 'sart_rts' not in st.session_state:
    st.session_state.sart_rts = []
if 'sart_mean_rt' not in st.session_state:
    st.session_state.sart_mean_rt = 0
if 'sart_commission_errors' not in st.session_state:
    st.session_state.sart_commission_errors = 0
if 'sart_omission_errors' not in st.session_state:
    st.session_state.sart_omission_errors = 0
if 'sart_start_time' not in st.session_state:
    st.session_state.sart_start_time = 0
if 'sart_is_nogo' not in st.session_state:
    st.session_state.sart_is_nogo = False

# N-Back State
if 'nback_completed' not in st.session_state:
    st.session_state.nback_completed = False
if 'nback_trial' not in st.session_state:
    st.session_state.nback_trial = 0
if 'nback_sequence' not in st.session_state:
    st.session_state.nback_sequence = []
if 'nback_current_letter' not in st.session_state:
    st.session_state.nback_current_letter = None
if 'nback_can_respond' not in st.session_state:
    st.session_state.nback_can_respond = False
if 'nback_responses' not in st.session_state:
    st.session_state.nback_responses = []
if 'nback_rts' not in st.session_state:
    st.session_state.nback_rts = []
if 'nback_mean_rt' not in st.session_state:
    st.session_state.nback_mean_rt = 0
if 'nback_accuracy' not in st.session_state:
    st.session_state.nback_accuracy = 0
if 'nback_start_time' not in st.session_state:
    st.session_state.nback_start_time = 0

# General State
if 'nlp_text' not in st.session_state:
    st.session_state.nlp_text = ""
if 'results' not in st.session_state:
    st.session_state.results = None

# ============================================================================
# MODEL LOADING
# ============================================================================
@st.cache_resource
def load_pvt_model():
    try:
        model_path = "pvt_lstm_model.h5"
        if os.path.exists(model_path):
            return load_model(model_path)
        return None
    except Exception as e:
        return None

@st.cache_resource
def load_sart_nback_model():
    try:
        model_path = "sart_2back_rf_model.pkl"
        if os.path.exists(model_path):
            return joblib.load(model_path)
        return None
    except Exception as e:
        return None

@st.cache_resource
def load_nlp_model():
    try:
        model_path = "./saved_models/bert_burnout_model"
        if os.path.exists(model_path):
            tokenizer = BertTokenizer.from_pretrained(model_path)
            model = BertForSequenceClassification.from_pretrained(model_path)
            return tokenizer, model
        return None, None
    except Exception as e:
        return None, None

pvt_model = load_pvt_model()
sart_nback_model = load_sart_nback_model()
nlp_tokenizer, nlp_model = load_nlp_model()

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def get_risk_level(score):
    if score <= 0.3:
        return "Low", "L"
    elif score <= 0.69:
        return "Medium", "M"
    else:
        return "High", "H"

def predict_pvt_lapse(rt_sequence, window_size=10):
    if pvt_model is None or len(rt_sequence) < window_size:
        return 0.5
    try:
        windows = []
        for i in range(len(rt_sequence) - window_size):
            window = rt_sequence[i:(i + window_size)]
            windows.append(window)
        if len(windows) == 0: return 0.5
        X = np.array(windows[-1]).reshape(1, window_size, 1)
        prediction = pvt_model.predict(X, verbose=0)[0][0]
        return float(prediction)
    except:
        return 0.5

def predict_sart_nback(sart_mean_rt, commission_errors, nback_mean_rt, accuracy):
    if sart_nback_model is None:
        return 0.5
    try:
        features = np.array([[sart_mean_rt, commission_errors, nback_mean_rt, accuracy]])
        prediction = sart_nback_model.predict_proba(features)[0][1]
        return float(prediction)
    except:
        return 0.5

def predict_nlp_sentiment(text):
    if nlp_model is None or nlp_tokenizer is None or not text.strip():
        return 0.5
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inputs = nlp_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=64).to(device)
        with torch.no_grad():
            outputs = nlp_model(**inputs)
            score = outputs.logits.item()
        score = torch.sigmoid(outputs.logits).item() 
        return float(score)
    except:
        return 0.5

def fuse_predictions(pvt_score, sart_nback_score, nlp_score, weights=(0.35, 0.35, 0.30)):
    return (pvt_score * weights[0] + sart_nback_score * weights[1] + nlp_score * weights[2])

# ============================================================================
# GAME FUNCTIONS
# ============================================================================
def run_pvt_task():
    st.subheader(" PVT - Reaction Time Test")
    st.markdown("*Click the button as quickly as possible when it turns RED*")
    
    num_trials = 15

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        status_text = st.empty()
        display_box = st.empty()
        rt_text = st.empty()
        
        is_disabled = not st.session_state.pvt_can_click
        click_button = st.button(" CLICK HERE ", key="pvt_click_btn", use_container_width=True, disabled=is_disabled)
    
    if st.session_state.pvt_trial == 0 and not st.session_state.pvt_show_stimulus and not st.session_state.pvt_completed:
        if st.button(" Start PVT Test", key="pvt_start_btn"):
            st.session_state.pvt_trial = 1
            st.session_state.pvt_rts = []
            st.rerun()
    
    if st.session_state.pvt_trial > 0 and st.session_state.pvt_trial <= num_trials:
        if not st.session_state.pvt_show_stimulus:
            status_text.text(f"Trial {st.session_state.pvt_trial}/{num_trials} - Wait...")
            display_box.markdown("""
                <div class="game-container"><div style='text-align:center; color:white; font-size:24px;'>Get Ready...</div></div>
            """, unsafe_allow_html=True)
            time.sleep(random.uniform(2, 5))
            st.session_state.pvt_show_stimulus = True
            st.session_state.pvt_can_click = True
            st.session_state.pvt_start_time = time.time()
            st.rerun()
        else:
            status_text.text(f"Trial {st.session_state.pvt_trial}/{num_trials} - CLICK!")
            display_box.markdown("""
                <div class="game-container" style='background-color: #dc2626;'>
                    <div style='text-align:center; color:white; font-size:48px; font-weight:bold;'>CLICK!</div>
                </div>
            """, unsafe_allow_html=True)
            
            if click_button:
                rt = time.time() - st.session_state.pvt_start_time
                st.session_state.pvt_rts.append(rt)
                rt_text.text(f"RT: {rt*1000:.0f}ms")
                
                st.session_state.pvt_can_click = False
                st.session_state.pvt_show_stimulus = False
                st.session_state.pvt_trial += 1
                st.rerun()

    elif st.session_state.pvt_trial > num_trials:
        st.session_state.pvt_completed = True
        status_text.text("PVT Complete!")
        display_box.markdown("""
            <div class="game-container" style='background-color: #059669;'>
                <div style='text-align:center; color:white; font-size:32px;'>✅ Complete!</div>
            </div>
        """, unsafe_allow_html=True)
        
        if len(st.session_state.pvt_rts) > 0:
            mean_rt = np.mean(st.session_state.pvt_rts) * 1000
            lapses = sum(1 for rt in st.session_state.pvt_rts if rt > 0.25)
            st.metric("Mean Reaction Time", f"{mean_rt:.0f}ms")
            st.metric("Lapses (>250ms)", lapses)

def run_sart_task():
    st.subheader(" SART - Inhibitory Control Test")
    st.markdown("*Click for every number **EXCEPT 3**. Do NOT click when you see 3!*")
    
    num_trials = 15
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        status_text = st.empty()
        display_box = st.empty()
        score_text = st.empty()
        
        is_disabled = not st.session_state.sart_can_respond
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            click_btn = st.button("✅ Click", key="sart_click_btn", use_container_width=True, disabled=is_disabled)
        with col_btn2:
            skip_btn = st.button(" Skip", key="sart_skip_btn", use_container_width=True, disabled=is_disabled)
    
    if st.session_state.sart_trial == 0 and st.session_state.sart_current_number is None and not st.session_state.sart_completed:
        if st.button(" Start SART Test", key="sart_start_btn"):
            st.session_state.sart_trial = 1
            st.session_state.sart_responses = []
            st.session_state.sart_rts = []
            st.rerun()
    
    if st.session_state.sart_trial > 0 and st.session_state.sart_trial <= num_trials:
        if st.session_state.sart_current_number is None:
            is_nogo = random.random() < 0.2
            number = 3 if is_nogo else random.choice([1, 2, 4, 5, 6, 7, 8, 9])
            st.session_state.sart_current_number = number
            st.session_state.sart_is_nogo = is_nogo
            st.session_state.sart_start_time = time.time()
            st.session_state.sart_can_respond = True
            st.rerun()
        else:
            status_text.text(f"Trial {st.session_state.sart_trial}/{num_trials}")
            color = "#dc2626" if st.session_state.sart_is_nogo else "#1e3a8a"
            display_box.markdown(f"""
                <div class="game-container" style='background-color: {color};'>
                    <div style='text-align: center; color: white; font-size: 120px; font-weight: bold;'>
                        {st.session_state.sart_current_number}
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            # Track BOTH error types in real-time
            commission_errors = sum(1 for r in st.session_state.sart_responses if r == "commission")
            omission_errors = sum(1 for r in st.session_state.sart_responses if r == "omission")
            score_text.text(f"Commission: {commission_errors} | Omission: {omission_errors}")
            
            if click_btn:
                rt = time.time() - st.session_state.sart_start_time
                st.session_state.sart_rts.append(rt)
                if st.session_state.sart_is_nogo:
                    st.session_state.sart_responses.append("commission")  # Clicked on 3 (ERROR)
                else:
                    st.session_state.sart_responses.append("correct_go")  # Clicked on non-3 (CORRECT)
                
                st.session_state.sart_can_respond = False
                st.session_state.sart_current_number = None
                st.session_state.sart_trial += 1
                st.rerun()
            elif skip_btn:
                if st.session_state.sart_is_nogo:
                    st.session_state.sart_responses.append("correct_inhibit")  # Skipped 3 (CORRECT)
                else:
                    st.session_state.sart_responses.append("omission")  # Skipped non-3 (ERROR)
                
                st.session_state.sart_can_respond = False
                st.session_state.sart_current_number = None
                st.session_state.sart_trial += 1
                st.rerun()

    elif st.session_state.sart_trial > num_trials:
        st.session_state.sart_completed = True
        status_text.text("SART Complete!")
        display_box.markdown("""
            <div class="game-container" style='background-color: #059669;'>
                <div style='text-align: center; color: white; font-size: 32px;'>✅ Complete!</div>
            </div>
        """, unsafe_allow_html=True)
        
        # Calculate final metrics
        commission_errors = sum(1 for r in st.session_state.sart_responses if r == "commission")
        omission_errors = sum(1 for r in st.session_state.sart_responses if r == "omission")
        mean_rt = np.mean(st.session_state.sart_rts) * 1000 if st.session_state.sart_rts else 0
        
        st.session_state.sart_mean_rt = mean_rt
        st.session_state.sart_commission_errors = commission_errors
        st.session_state.sart_omission_errors = omission_errors
        
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric("Mean RT", f"{mean_rt:.0f}ms")
            st.metric("Commission Errors", commission_errors, help="Clicked on 3 (should have skipped)")
        with col_m2:
            st.metric("Omission Errors", omission_errors, help="Skipped non-3 (should have clicked)")

def run_nback_task():
    st.subheader("2-Back - Working Memory Test")
    st.markdown("*Click **MATCH** if current letter matches the one from **2 steps ago***")
    
    num_trials = 15
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        status_text = st.empty()
        display_box = st.empty()
        history_text = st.empty()
        accuracy_text = st.empty()
        
        is_disabled = not st.session_state.nback_can_respond
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            match_btn = st.button(" MATCH", key="nback_match_btn", use_container_width=True, disabled=is_disabled)
        with col_btn2:
            nomatch_btn = st.button(" NO MATCH", key="nback_nomatch_btn", use_container_width=True, disabled=is_disabled)
    
    if st.session_state.nback_trial == 0 and not st.session_state.nback_sequence and not st.session_state.nback_completed:
        if st.button(" Start 2-Back Test", key="nback_start_btn"):
            st.session_state.nback_trial = 1
            st.session_state.nback_sequence = [random.choice(letters) for _ in range(num_trials)]
            for i in range(2, num_trials):
                if random.random() < 0.3:
                    st.session_state.nback_sequence[i] = st.session_state.nback_sequence[i-2]
            st.session_state.nback_responses = []
            st.session_state.nback_rts = []
            st.rerun()
    
    if st.session_state.nback_trial > 0 and st.session_state.nback_trial <= num_trials:
        if st.session_state.nback_current_letter is None:
            idx = st.session_state.nback_trial - 1
            st.session_state.nback_current_letter = st.session_state.nback_sequence[idx]
            st.session_state.nback_start_time = time.time()
            st.session_state.nback_can_respond = True
            st.rerun()
        else:
            status_text.text(f"Trial {st.session_state.nback_trial}/{num_trials}")
            display_box.markdown(f"""
                <div class="game-container" style='background-color: #1e3a8a;'>
                    <div style='text-align: center; color: white; font-size: 100px; font-weight: bold;'>
                        {st.session_state.nback_current_letter}
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            idx = st.session_state.nback_trial - 1
            if idx >= 2:
                history_text.text(f"2-back: {st.session_state.nback_sequence[idx-2]} | 1-back: {st.session_state.nback_sequence[idx-1]}")
            else:
                history_text.text("Building sequence...")
            
            hits = sum(1 for r in st.session_state.nback_responses if r == "hit")
            total_targets = sum(1 for r in st.session_state.nback_responses if r in ["hit", "miss"])
            accuracy = (hits / total_targets * 100) if total_targets > 0 else 0
            accuracy_text.text(f"Accuracy: {accuracy:.0f}%")
            
            is_target = (idx >= 2 and st.session_state.nback_current_letter == st.session_state.nback_sequence[idx-2])
            
            if match_btn:
                rt = time.time() - st.session_state.nback_start_time
                st.session_state.nback_rts.append(rt)
                if is_target:
                    st.session_state.nback_responses.append("hit")
                else:
                    st.session_state.nback_responses.append("false_alarm")
                
                st.session_state.nback_can_respond = False
                st.session_state.nback_current_letter = None
                st.session_state.nback_trial += 1
                st.rerun()
            elif nomatch_btn:
                if is_target:
                    st.session_state.nback_responses.append("miss")
                else:
                    st.session_state.nback_responses.append("correct_rejection")
                
                st.session_state.nback_can_respond = False
                st.session_state.nback_current_letter = None
                st.session_state.nback_trial += 1
                st.rerun()

    elif st.session_state.nback_trial > num_trials:
        st.session_state.nback_completed = True
        status_text.text("2-Back Complete!")
        display_box.markdown("""
            <div class="game-container" style='background-color: #059669;'>
                <div style='text-align: center; color: white; font-size: 32px;'>✅ Complete!</div>
            </div>
        """, unsafe_allow_html=True)
        
        hits = sum(1 for r in st.session_state.nback_responses if r == "hit")
        misses = sum(1 for r in st.session_state.nback_responses if r == "miss")
        false_alarms = sum(1 for r in st.session_state.nback_responses if r == "false_alarm")
        correct_rejections = sum(1 for r in st.session_state.nback_responses if r == "correct_rejection")
        
        total_targets = hits + misses
        overall_accuracy = ((hits + correct_rejections) / num_trials * 100) if num_trials > 0 else 0
        mean_rt = np.mean(st.session_state.nback_rts) * 1000 if st.session_state.nback_rts else 0
        
        st.session_state.nback_mean_rt = mean_rt
        st.session_state.nback_accuracy = overall_accuracy
        
        st.metric("Accuracy", f"{overall_accuracy:.0f}%")
        st.metric("Hits", f"{hits}/{total_targets}")
        st.metric("Mean RT", f"{mean_rt:.0f}ms")

# ============================================================================
# PAGE RENDERING
# ============================================================================
def render_home():
    st.title("Burnout Risk Predictor")
    st.markdown("---")
    st.markdown("""
    ### Welcome to Your Burnout Assessment
    This tool uses **three cognitive tasks** and **sentiment analysis** to provide an objective assessment of your burnout risk level.
    
    **What to expect:**
      **PVT**: Reaction time test
      **SART**: Inhibitory control test  
      **2-Back**: Working memory test
      **NLP**: Share how you're feeling
    
    **Total Time:** ~7-10 minutes
    """)
    if st.button("Start Assessment", type="primary", use_container_width=True):
        st.session_state.current_page = 1
        st.rerun()

def render_tasks():
    st.title("Cognitive Assessment Tasks")
    st.markdown("Complete all three tasks to proceed.")
    st.markdown("---")
    
    with st.expander("Task 1: PVT", expanded=True):
        run_pvt_task()
        if st.session_state.pvt_completed:
            st.success("✅ PVT Completed!")
    
    st.markdown("---")
    
    with st.expander("Task 2: SART", expanded=False):
        run_sart_task()
        if st.session_state.sart_completed:
            st.success("✅ SART Completed!")
    
    st.markdown("---")
    
    with st.expander("Task 3: 2-Back", expanded=False):
        run_nback_task()
        if st.session_state.nback_completed:
            st.success("✅ 2-Back Completed!")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬅Back to Home", use_container_width=True):
            st.session_state.current_page = 0
            st.rerun()
    
    with col2:
        if st.session_state.pvt_completed and st.session_state.sart_completed:
            if st.button("Proceed to NLP Input", type="primary", use_container_width=True):
                st.session_state.current_page = 2
                st.rerun()
        else:
            st.info(" Complete PVT and SART tasks to proceed")

def render_nlp():
    st.title("How Are You Feeling?")
    st.markdown("Share your thoughts about your current mental and emotional state.")
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        nlp_input = st.text_area("Describe how you've been feeling lately", value=st.session_state.nlp_text, height=200)
    with col2:
        st.info("**Tips:**\n- Be honest\n- Mention stress levels")
    
    st.session_state.nlp_text = nlp_input
    
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Back", use_container_width=True):
            st.session_state.current_page = 1
            st.rerun()
    with col2:
        if st.button("Save & Continue", type="primary", use_container_width=True):
            if len(nlp_input) < 20:
                st.warning("Please provide more text")
            else:
                st.session_state.current_page = 3
                st.rerun()
    with col3:
        if st.button(" Skip", use_container_width=True):
            st.session_state.nlp_text = "No text provided"
            st.session_state.current_page = 3
            st.rerun()

def render_results():
    st.title("Your Burnout Risk Assessment Results")
    st.markdown("---")
    
    # VALIDATION: Check if tasks were actually completed
    if not st.session_state.pvt_completed or not st.session_state.sart_completed:
        st.error("**Incomplete Assessment!**")
        st.warning("""
        You haven't completed all the required tasks yet.
        
        **Required:**
          PVT Task (Reaction Time)
          SART Task (Inhibitory Control)
        
        **Optional:**
          NLP Text Input
        """)
        
        st.info("Please complete the cognitive tasks first to get accurate results.")
        
        if st.button(" Go to Cognitive Tasks", type="primary", use_container_width=True):
            st.session_state.current_page = 1
            st.rerun()
        return  # Exit the function early, don't show results
    
    # Calculate results only if not already done
    if st.session_state.results is None:
        with st.spinner(" Analyzing your data with AI models..."):
            time.sleep(2)
            
            pvt_rts = st.session_state.pvt_rts
            sart_mean_rt = st.session_state.sart_mean_rt
            sart_commission_errors = st.session_state.sart_commission_errors
            nback_mean_rt = st.session_state.nback_mean_rt
            nback_accuracy = st.session_state.nback_accuracy
            nlp_text = st.session_state.nlp_text
            
            # Additional validation: Check if we have actual data
            if len(pvt_rts) == 0:
                st.error(" No PVT data recorded. Please complete the PVT task.")
                return
            
            if sart_mean_rt == 0 and sart_commission_errors == 0:
                st.error(" No SART data recorded. Please complete the SART task.")
                return
            
            pvt_score = predict_pvt_lapse(pvt_rts) if pvt_rts else 0.5
            sart_nback_score = predict_sart_nback(sart_mean_rt, sart_commission_errors, nback_mean_rt, nback_accuracy)
            nlp_score = predict_nlp_sentiment(nlp_text)
            fused_score = fuse_predictions(pvt_score, sart_nback_score, nlp_score)
            risk_level, risk_emoji = get_risk_level(fused_score)
            
            st.session_state.results = {
                'pvt_score': pvt_score,
                'sart_nback_score': sart_nback_score,
                'nlp_score': nlp_score,
                'fused_score': fused_score,
                'risk_level': risk_level,
                'risk_emoji': risk_emoji,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
    
    results = st.session_state.results
    
    # Main Risk Display
    st.markdown(f"""
    <div style='text-align: center; padding: 40px; background-color: {"#fee2e2" if results['risk_level'] == "High" else "#fef3c7" if results['risk_level'] == "Medium" else "#d1fae5"}; 
                border-radius: 15px; margin-bottom: 30px;'>
        <h1 style='font-size: 72px; margin: 0;'>{results['risk_emoji']}</h1>
        <h2 style='color: {"#dc2626" if results['risk_level'] == "High" else "#f59e0b" if results['risk_level'] == "Medium" else "#059669"}; 
                   margin: 10px 0;'>{results['risk_level'].upper()} BURNOUT RISK</h2>
        <p style='font-size: 24px; color: #4b5563;'>Overall Score: {results['fused_score']*100:.1f}%</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Individual Model Predictions
    st.subheader(" Individual Model Predictions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        pvt_level, _ = get_risk_level(results['pvt_score'])
        st.metric(
            label=" PVT Model (LSTM)",
            value=f"{results['pvt_score']*100:.1f}%",
            delta=f"{pvt_level} Risk"
        )
        st.progress(results['pvt_score'])
        st.caption("Analyzes reaction time patterns for fatigue indicators")
    
    with col2:
        sart_level, _ = get_risk_level(results['sart_nback_score'])
        st.metric(
            label=" SART+2-Back Model (RF)",
            value=f"{results['sart_nback_score']*100:.1f}%",
            delta=f"{sart_level} Risk"
        )
        st.progress(results['sart_nback_score'])
        st.caption("Evaluates attention control and working memory")
    
    with col3:
        nlp_level, _ = get_risk_level(results['nlp_score'])
        st.metric(
            label=" NLP Model (BERT)",
            value=f"{results['nlp_score']*100:.1f}%",
            delta=f"{nlp_level} Risk"
        )
        st.progress(results['nlp_score'])
        st.caption("Analyzes emotional sentiment from your text")
    
    st.markdown("---")
    
    # Model Performance Comparison
    st.subheader(" Model Performance Metrics")
    
    metrics_data = {
        'Model': ['PVT (LSTM)', 'SART+2-Back (RF)', 'NLP (BERT)'],
        'Accuracy': ['64.77%', '81.00%', 'N/A'],
        'F1-Score': ['N/A*', '0.79', 'N/A'],
        'Type': ['Deep Learning', 'Machine Learning', 'Transfer Learning'],
        'Your Score': [f"{results['pvt_score']*100:.1f}%", f"{results['sart_nback_score']*100:.1f}%", f"{results['nlp_score']*100:.1f}%"]
    }
    
    metrics_df = pd.DataFrame(metrics_data)
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    
    st.caption("*PVT uses probability-based LSTM output for continuous fatigue patterns. F1-Score requires hard binary predictions and ground truth labels, which are not available during real-time inference.")

    # Visualization
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(" Prediction Distribution")
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=['PVT', 'SART+2B', 'NLP', 'Fused'],
            y=[results['pvt_score']*100, results['sart_nback_score']*100, results['nlp_score']*100, results['fused_score']*100],
            marker_color=['#3b82f6', '#8b5cf6', '#10b981', '#f59e0b'],
            text=[f"{results['pvt_score']*100:.1f}%", f"{results['sart_nback_score']*100:.1f}%", 
                  f"{results['nlp_score']*100:.1f}%", f"{results['fused_score']*100:.1f}%"],
            textposition='auto'
        ))
        
        fig.update_layout(
            yaxis_title="Risk Score (%)",
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader(" Recommendations")
        if results['risk_level'] == "High":
            st.error("""
            ** High Risk Detected:**
            
            1. **Take a Break**: Consider taking time off
            2. **Seek Support**: Talk to a professional
            3. **Sleep Priority**: Aim for 7-9 hours
            4. **Reduce Workload**: Delegate tasks
            5. **Physical Activity**: Light exercise
            """)
        elif results['risk_level'] == "Medium":
            st.warning("""
            ** Moderate Risk:**
            
            1. **Monitor Stress**: Track daily
            2. **Work-Life Balance**: Set boundaries
            3. **Mindfulness**: Practice meditation
            4. **Social Connection**: Spend time with loved ones
            5. **Regular Breaks**: Take breaks every hour
            """)
        else:
            st.success("""
            ** Low Risk:**
            
            1. **Continue Good Practices**: Keep it up
            2. **Stay Active**: Regular exercise
            3. **Sleep Hygiene**: Prioritize sleep
            4. **Regular Check-ins**: Re-assess monthly
            5. **Build Resilience**: Develop coping strategies
            """)
    
    st.markdown("---")
    
    # Download Report
    st.subheader(" Download Your Report")
    
    report_data = {
        'Assessment Date': results['timestamp'],
        'Overall Risk Level': results['risk_level'],
        'Overall Score': f"{results['fused_score']*100:.2f}%",
        'PVT Score': f"{results['pvt_score']*100:.2f}%",
        'SART+2-Back Score': f"{results['sart_nback_score']*100:.2f}%",
        'NLP Score': f"{results['nlp_score']*100:.2f}%",
        'Mean Reaction Time (PVT)': f"{np.mean(st.session_state.pvt_rts)*1000:.0f}ms" if st.session_state.pvt_rts else "N/A",
        'Commission Errors (SART)': st.session_state.sart_commission_errors,
        'Omission Errors (SART)': st.session_state.get('sart_omission_errors', 0),
        '2-Back Accuracy': f"{st.session_state.nback_accuracy:.1f}%"
    }
    
    report_df = pd.DataFrame(list(report_data.items()), columns=['Metric', 'Value'])
    csv = report_df.to_csv(index=False).encode('utf-8')
    
    st.download_button(
        label=" Download CSV Report",
        data=csv,
        file_name=f"burnout_assessment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True
    )
    
    # Navigation
    col1, col2 = st.columns(2)
    with col1:
        if st.button(" Start New Assessment", use_container_width=True):
            # Reset session state
            st.session_state.pvt_rts = []
            st.session_state.pvt_completed = False
            st.session_state.pvt_trial = 0
            st.session_state.pvt_show_stimulus = False
            st.session_state.pvt_can_click = False
            
            st.session_state.sart_completed = False
            st.session_state.sart_trial = 0
            st.session_state.sart_current_number = None
            st.session_state.sart_can_respond = False
            st.session_state.sart_responses = []
            st.session_state.sart_rts = []
            st.session_state.sart_mean_rt = 0
            st.session_state.sart_commission_errors = 0
            st.session_state.sart_omission_errors = 0
            
            st.session_state.nback_completed = False
            st.session_state.nback_trial = 0
            st.session_state.nback_sequence = []
            st.session_state.nback_current_letter = None
            st.session_state.nback_can_respond = False
            st.session_state.nback_responses = []
            st.session_state.nback_rts = []
            st.session_state.nback_mean_rt = 0
            st.session_state.nback_accuracy = 0
            
            st.session_state.nlp_text = ""
            st.session_state.results = None
            st.session_state.current_page = 0
            st.rerun()
    
    with col2:
        st.info(" You can retake the assessment anytime")

def main():
    with st.sidebar:
        st.title(" Navigation")
        
        page = st.radio(
            "Go to:",
            ["Home", "Cognitive Tasks", "NLP Input", "Results Dashboard"],
            index=st.session_state.current_page,
            label_visibility="collapsed"
        )
        
        page_map = {
            "Home": 0,
            "Cognitive Tasks": 1,
            "NLP Input": 2,
            "Results Dashboard": 3
        }
        
        if page_map[page] != st.session_state.current_page:
            st.session_state.current_page = page_map[page]
            st.rerun()
        
        st.markdown("---")
        st.markdown("""
        ### About This Tool
        
        This multimodal burnout predictor combines:
         **Cognitive performance tasks** (PVT, SART, 2-Back)
         **Deep learning models** (LSTM for temporal patterns)
         **NLP sentiment analysis** (BERT-based emotional detection)
        
        **Total Time:** 7-10 minutes
        
        **Privacy:** All data is processed locally on your device. No personal information is stored or shared.
        
        ---
        
        ###  Based on Research
        
        - OpenNeuro Sleepy Brain Project
        - GoEmotions Dataset (Google Research)
        
        
        ---
        
        ###  Data Privacy
        
        Your cognitive performance data and text responses are processed in real-time and are **never** stored on any server. All analysis happens locally in your browser session.
        """)
    
    # Route to appropriate page
    if st.session_state.current_page == 0:
        render_home()
    elif st.session_state.current_page == 1:
        render_tasks()
    elif st.session_state.current_page == 2:
        render_nlp()
    elif st.session_state.current_page == 3:
        render_results()

if __name__ == "__main__":
    main()