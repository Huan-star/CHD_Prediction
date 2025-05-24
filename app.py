import streamlit as st
import os
import joblib
import pandas as pd
import numpy as np
import seaborn as sns
import shap
import matplotlib.pyplot as plt

# Set page configuration
st.set_page_config(
    page_title="Coronary Heart Disease Risk Prediction System",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://cardiology-help.com   ',
        'Report a bug': "https://cardiology-bug.com   ",
        'About': "### Coronary Heart Disease Risk Prediction System v1.0\nA machine learning-based cardiovascular disease risk assessment tool"
    }
)


def load_artifacts():
    current_dir = os.path.dirname(__file__)
    model_path = os.path.join(current_dir, 'stacking_classifier.pkl')
    scaler_path = os.path.join(current_dir, 'minmax_scaler.pkl')
    background_path = os.path.join(current_dir, 'background_data.pkl')

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    background = joblib.load(background_path)

    explainer = shap.KernelExplainer(model.predict_proba, background)
    
    return model, scaler, explainer, background

model, scaler, explainer, background = load_artifacts()
    

# Your ten features
features = ['sysBP', 'age', 'totChol', 'diaBP', 'glucose', 'cigsPerDay', 'prevalentHyp', 'BPMeds', 'BMI', 'diabetes']

# Chinese feature name mapping
feature_names_cn = {
    'sysBP': 'Systolic Blood Pressure (mmHg)',
    'age': 'Age',
    'glucose': 'Fasting Blood Sugar (mg/dL)',
    'cigsPerDay': 'Daily Cigarette Consumption (cigarette)',
    'totChol': 'Total Cholesterol (mg/dL)',
    'diaBP': 'Diastolic Blood Pressure (mmHg)',
    'prevalentHyp': 'History of Hypertension (0=No, 1=Yes)',
    'diabetes': 'diabetes (0=No, 1=Yes)',
    'BPMeds': 'Use of Antihypertensive Medication (0=No, 1=Yes)',
    'BMI': 'BMI'
}

# Normal range references (example values, can be adjusted according to medical standards)
normal_ranges = {
    'age': (0, 60),             # Age range
    'sysBP': (120, 130),        # Normal systolic blood pressure range
    'diaBP': (80, 90),          # Normal diastolic blood pressure range
    'totChol': (0, 200),        # Normal total cholesterol range
    'glucose': (70, 126),       # Normal fasting blood sugar range
    'cigsPerDay': (0, 0),       # Ideal value is 0 (non-smoker)
    'BMI': (18.5, 24.9),        # Normal BMI range
    'diabetes': (0, 0),          # 0=No history of diabetes
    'prevalentHyp': (0, 0),     # 0=No history of hypertension
    'BPMeds': (0, 0)           # 0=Not using antihypertensive medication (ideal value)
}

# Background data for all user inputs
all_background_data = []

# Page header and introduction
def render_header():
    col1, col2 = st.columns([1, 3])
    with col1:
        st.subheader("❤️")
    with col2:
        st.title("Coronary Heart Disease Risk Prediction System")
        st.markdown("""
        <style>
       .big-font {
            font-size:16px !important;
            color: #2c3e50;
        }
       .header-box {
            background-color: #f8f9fa;
            border-radius: 10px;
            padding: 1rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        </style>
        <div class="header-box">
            <div class="big-font">
            This system uses a machine learning model to predict the risk of coronary heart disease based on 10 key indicators.<br>
            Please enter the patient's latest health data to obtain a risk assessment.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

# Input form
def input_features():
    st.header("📋 Patient Health Indicator Input")
    st.markdown("Please enter the following health indicators related to coronary heart disease:")

    inputs = {}
    cols = st.columns(2)  # Display 10 features in two columns

    for i, feature in enumerate(features):
        with cols[i % 2]:
            min_val, max_val = normal_ranges[feature]
            # Handle binary feature input (0/1)
            if feature in ['prevalentHyp','diabetes', 'BPMeds']:
                inputs[feature] = st.radio(
                    f"**{feature_names_cn[feature]}**",
                    options=[0, 1],
                    help=f"0=No, 1=Yes",
                    key=f"radio_{feature}"
                )
            else:
                # Calculate a reasonable default value
                default_value = float((min_val + max_val) / 2)
                if default_value > max_val:
                    default_value = max_val
                elif default_value < min_val:
                    default_value = min_val

                inputs[feature] = st.number_input(
                    f"**{feature_names_cn[feature]}**",
                    min_value=0.0,
                    max_value=400.0,
                    value=default_value,
                    step=0.1,
                    help=f"Normal range: {min_val}-{max_val}",
                    key=f"input_{feature}"
                )
    return inputs

# Display input values visualization
def display_input_visualization(inputs):
    st.subheader("🔮 SHAP Force Plot After Input")

    input_data = pd.DataFrame([inputs])
    input_data_scaled = scaler.transform(input_data.copy())

    shap_values = explainer.shap_values(input_data_scaled)

    if isinstance(shap_values, list) and len(shap_values) > 1:
        shap_values_single = shap_values[1]
    else:
        shap_values_single = shap_values

    shap_values_class1 = shap_values_single[0, :, 1]

    if len(shap_values_class1) != len(features):
        raise ValueError(f"The length of the SHAP value（{len(shap_values_class1)}）with the number of features({len(features)})mismatching!")

    sample_explanation = shap.Explanation(
        values=shap_values_class1,
        base_values=explainer.expected_value[1],
        data=input_data_scaled[0],
        feature_names=features
    )

    plt.figure(figsize=(15, 8))

    shap.force_plot(
        sample_explanation.base_values,
        sample_explanation.values,
        sample_explanation.data,
        feature_names=features,
        matplotlib=True,
        text_rotation=15,
    )
    plt.tight_layout()

    from io import BytesIO
    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    plt.close()

    import base64
    img_str = base64.b64encode(buffer.getvalue()).decode()

    st.markdown(
        f"""
        <style>
            .shap-plot {{
                width: 90%;
                height: auto;
                margin: 0 auto;
                display: block;
            }}
        </style>
        <img src="data:image/png;base64,{img_str}" class="shap-plot">
        """,
        unsafe_allow_html=True
    )
    st.header("📊 Input Indicator Visualization Analysis")

    # Convert to DataFrame
    df = pd.DataFrame.from_dict(inputs, orient='index', columns=['Value'])
    df['Indicator'] = [feature_names_cn[f] for f in df.index]
    df['Normal Range Lower'] = [normal_ranges[f][0] for f in df.index]
    df['Normal Range Upper'] = [normal_ranges[f][1] for f in df.index]

    # Initialize 'Deviation' column
    df['Deviation'] = 0.0  # Initial value is 0.0

    # Determine abnormality and calculate deviation
    def is_abnormal(row):
        prevalent_hyp_value = inputs.get('prevalentHyp', 0)
        bp_meds_value = inputs.get('BPMeds', 0)

        if row.name in ['prevalentHyp','diabetes', 'BPMeds']:
            if row.name == 'prevalentHyp':
                return prevalent_hyp_value == 1
            elif row.name == 'BPMeds':
                return prevalent_hyp_value == 1 and bp_meds_value == 0
            else:
                return row['Value'] not in [0, 1]
        else:
            # Calculate deviation
            if row['Value'] > row['Normal Range Upper']:
                df.loc[row.name, 'Deviation'] = row['Value'] - row['Normal Range Upper']
            elif row['Value'] < row['Normal Range Lower']:
                df.loc[row.name, 'Deviation'] = row['Normal Range Lower'] - row['Value']
            return (row['Value'] < row['Normal Range Lower']) | (row['Value'] > row['Normal Range Upper'])

    df['Is Abnormal'] = df.apply(is_abnormal, axis=1)

    # Display data table
    st.subheader("📝 Input Indicator Summary")

    # Ensure display_df includes 'Is Abnormal' and 'Deviation' columns
    display_df = df[['Indicator', 'Value', 'Normal Range Lower', 'Normal Range Upper', 'Is Abnormal', 'Deviation']].copy()
    display_df.columns = ['Health Indicator', 'Current Value', 'Normal Range (Low)', 'Normal Range (High)', 'Is Abnormal', 'Deviation']

    # Format the Deviation column
    display_df['Deviation'] = display_df['Deviation'].apply(lambda x: f"{x:.2f}" if x != 0 else "0")

    # Highlight abnormal indicators with styles
    def highlight_abnormal(row):
        if row['Is Abnormal']:
            return ['background: #ffdddd'] * len(row)  # Abnormal indicator background color is light red
        else:
            return [''] * len(row)  # Normal indicators do not change background color

    st.dataframe(
        display_df.style.apply(highlight_abnormal, axis=1),
        height=400,
        use_container_width=True
    )

    # Visualize abnormal indicators
    st.subheader("⚠️ Abnormal Indicator Alert")
    abnormal_count = df['Is Abnormal'].sum()

    if abnormal_count > 0:
        # Use card layout to display the number of abnormal indicators
        st.warning(f"**Found {abnormal_count} abnormal indicators!** Please review the detailed analysis below.")

        abnormal_df = df[df['Is Abnormal']].copy()

        # Create two-column layout
        col1, col2 = st.columns(2)

        with col1:
            # Abnormal indicator bar chart
            plt.figure(figsize=(5, 4))  # Adjust image size

            # Calculate deviation degree (normalization processing)
            max_deviation = abnormal_df['Deviation'].abs().max()
            if max_deviation > 0:
                deviation_normalized = abnormal_df['Deviation'].abs() / max_deviation
            else:
                deviation_normalized = abnormal_df['Deviation'].abs()

            sns.barplot(
                x=abnormal_df['Indicator'],
                y=deviation_normalized,  # Use normalized deviation degree
                palette='Oranges_r'
            )
            plt.title("Degree of Deviation from Normal Range for Abnormal Indicators")
            plt.xlabel("Indicator")
            plt.ylabel("Deviation Degree")
            plt.xticks(rotation=45)
            st.pyplot(plt.gcf())

        with col2:
            # Abnormal indicator radar chart
            st.info("**Abnormal Indicator Radar Chart**")
            fig = plt.figure(figsize=(5, 5))  # Adjust image size
            ax = fig.add_subplot(111, polar=True)

            categories = abnormal_df['Indicator'].tolist()
            values = abnormal_df['Deviation'].abs().tolist()
            num_vars = len(categories)

            # Calculate angles
            angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
            values += values[:1]
            angles += angles[:1]

            ax.plot(angles, values, color='red', linewidth=1)
            ax.fill(angles, values, color='red', alpha=0.25)
            ax.set_thetagrids(np.degrees(angles[:-1]), categories)
            ax.set_title("Distribution of Abnormal Indicators", size=10, y=1.1)
            st.pyplot(fig)
    else:
        st.success("✅ All indicators are within the normal range")

    st.markdown("---")
    return df

# Prediction and display
def predict_and_display(inputs, input_df):
    st.header("🔍 Coronary Heart Disease Risk Prediction Results")

    # Convert to model input format
    input_data = pd.DataFrame([inputs])
    input_data_scaled = scaler.transform(input_data.copy())

    # Prediction
    prediction = model.predict(input_data_scaled)[0]
    proba = model.predict_proba(input_data_scaled)[0][0]  # Get positive probability

    # Two-column layout
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("❗ Risk Level")

        # Check for high-risk based on multiple significant abnormalities
        abnormal_df = input_df[input_df['Is Abnormal']].copy()
        high_deviation_df = abnormal_df[abnormal_df['Deviation'].abs() > 0.3]
        high_deviation_count = len(high_deviation_df)
        high_risk_abnormalities = prediction == 1

        if prediction == 1:
            st.error(f"""
            ### ⚠️ High Risk Alert
            **High risk of coronary heart disease**
            """)
            proba = model.predict_proba(input_data_scaled)[0][1]
            st.markdown(f"""
            <div style="background-color:#f8f9fa;padding:20px;border-radius:10px;text-align:center;">
                <h3 style="color:#2c3e50;">Coronary Heart Disease Risk Probability</h3>
                <div style="font-size:36px;font-weight:bold;color:{'#e74c3c' if proba > 0.5 else '#27ae60'};">
                    {proba*100:.1f}%
                </div>
                <progress value="{proba}" max="1" style="width:100%;height:20px;"></progress>
            </div>
        """, unsafe_allow_html=True)

        else:
            st.success("""
            ### ✅ Low Risk Indication
            **Low risk of coronary heart disease**
            """)
            proba = model.predict_proba(input_data_scaled)[0][1]
            st.markdown(f"""
            <div style="background-color:#f8f9fa;padding:20px;border-radius:10px;text-align:center;">
                <h3 style="color:#2c3e50;">Coronary Heart Disease Risk Probability</h3>
                <div style="font-size:36px;font-weight:bold;color:{'#e74c3c' if proba > 0.5 else '#27ae60'};">
                    {proba*100:.1f}%
                </div>
                <progress value="{proba}" max="1" style="width:100%;height:20px;"></progress>
            </div>
        """, unsafe_allow_html=True)

        # If high-risk due to abnormalities, display the specific indicators
        if high_risk_abnormalities:
            abnormal_indicators = high_deviation_df['Indicator'].tolist()
            indicators_str = " or ".join(abnormal_indicators)
            st.markdown(f"""
            <div style="background-color:#fff3f3;padding:30px;border-radius:10px;border-left:5px solid #e74c3c;">
                <h4 style="color:#e74c3c;">High Deviation Indicators</h4>
                <p style="color:#2c3e50;">The following indicators have significant deviations: {indicators_str}</p>
            </div>
            """, unsafe_allow_html=True)


    with col2:
        # Health recommendations
        st.subheader("💡 Personalized Recommendations")

        show_high_risk = prediction == 1

        if show_high_risk:
            st.markdown("""
            <div style="background-color:#fff3f3;padding:15px;border-radius:10px;border-left:5px solid #e74c3c;">
                <h4 style="color:#e74c3c;">High Risk Management Measures</h4>
                <ul style="color:#2c3e50;">
                    <li>It is recommended to have an electrocardiogram (ECG), echocardiography (UCG) check within 3 months</li>
                    <li>Control blood pressure/blood sugar/blood lipids under the guidance of a doctor, and follow medical advice</li>
                    <li>Follow a low-fat, low-salt diet (daily salt intake <5g)</li>
                    <li>Avoid high-cholesterol foods and eat more fiber</li>
                    <li>Seek weight loss for obesity with a doctor's help</li>
                    <li>Quit smoking and limit alcohol intake</li>
                    <li>Maintain good sleep quality, at least 7 hours per night</li>
                    <li>Regularly monitor heart rate</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        else:
            st.markdown("""
            <div style="background-color:#f0fff4;padding:15px;border-radius:10px;border-left:5px solid #27ae60;">
                <h4 style="color:#27ae60;">Low Risk Maintenance Recommendations</h4>
                <ul style="color:#2c3e50;">
                    <li>Maintain a healthy lifestyle and regular health check-ups</li>
                    <li>Continue to control body weight (maintain BMI between 18.5-24.9)</li>
                    <li>Monitor blood pressure/blood sugar/cholesterol levels</li>
                    <li>Avoid long-term mental stress</li>
                    <li>Have a cardiovascular risk assessment once a year</li>
                    <li>Maintain a balanced diet, eat more vegetables, fruits, and whole grains</li>
                    <li>Maintain regular exercise habits</li>
                    <li>Maintain good mental health and actively deal with stress in life</li>
                    <li>Regularly communicate with your doctor and adjust your health management plan in a timely manner</li>
                    <li>Avoid overwork and arrange work and rest time reasonably</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

    # Disclaimer
    st.markdown("""
    <div style="background-color:#f8f9fa; padding:20px; border-radius:10px; text-align:center; width:100%; box-sizing:border-box; margin-top:20px;">
        <h3 style="color:#2c3e50;">📌 Important Notes</h3>
        <ul style="color:#2c3e50; list-style-type:none; padding:0;">
            <li>This prediction result is based on a statistical model and cannot replace professional medical diagnosis</li>
            <li>If the result is abnormal, please consult a cardiologist in a timely manner</li>
            <li>The model's performance may be affected by the quality of the input data</li>
            <li>It is recommended to make a comprehensive judgment in combination with family history and clinical examinations</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

# Main function
def main():
    # Custom CSS
    st.markdown("""
    <style>
    .stNumberInput .st-input {
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .stRadio .stRadio-option {
        border-radius: 8px;
        margin-right: 10px;
        padding: 8px 16px;
    }
    </style>
    """, unsafe_allow_html=True)

    render_header()
    inputs = input_features()
    input_df = display_input_visualization(inputs)

    if st.button("⚕️ Calculate Coronary Heart Disease Risk", type="primary", use_container_width=True):
        with st.spinner('Analyzing cardiovascular risk...'):
            # Add current input to background data
            all_background_data.append(inputs.copy())
            predict_and_display(inputs, input_df)

if __name__ == "__main__":
    main()
