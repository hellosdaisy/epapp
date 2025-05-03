import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

model = joblib.load('catboost.pkl')
scaler = joblib.load('scaler.pkl') 

# Define feature options
Vaginal_bleeding_options = {    
    0: 'None (0)',    
    1: 'Less (1)',    
    2: 'Equivalent (2)'
}

HCG_options = {    
    0: 'hCG＜1000 (0)',    
    1: '1000≤hCG＜2000 (1)',    
    2: '2000≤hCG＜3000 (2)',
    3: '3000≤hCG＜4000 (3)',    
    4: '4000≤hCG＜5000 (4)', 
    5: 'hCG≥5000 (5)'
}

# Define feature names
feature_names = [    
    "Gravidity", "Progesterone", "History of pelvic surgery", "History of cesarean section", "Abdominal tenderness",    
    "Homogeneous adnexal mass", "HCG", "Vaginal_bleeding"
]

# Streamlit user interface
st.title("Ectopic Pregnancy Risk Assessment")

# numerical input
Gravidity = st.number_input("Gravidity:", min_value=0, max_value=8, value=1)
Progesterone = st.number_input("Progesterone(ng/ml):", min_value=0.29, max_value=58.0, value=15.0)

#categorical selection
History_of_pelvic_surgery = st.selectbox("History_of_pelvic_surgery :", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')
History_of_cesarean_section = st.selectbox("History_of_cesarean_section :", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')
Abdominal_tenderness= st.selectbox("Abdominal_tenderness :", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')
Homogeneous_adnexal_mass = st.selectbox("Homogeneous_adnexal_mass:", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')
Vaginal_bleeding = st.selectbox("Vaginal_bleeding(compare with menstrual flow):", options=list(Vaginal_bleeding_options.keys()), format_func=lambda x: Vaginal_bleeding_options[x])
HCG = st.selectbox("hCG(mIU/ml):", options=list(HCG_options.keys()), format_func=lambda x: HCG_options[x])

# Process inputs and make predictions
feature_values = [Gravidity, History_of_pelvic_surgery, History_of_cesarean_section, Abdominal_tenderness, Vaginal_bleeding, Homogeneous_adnexal_mass, HCG,Progesterone]
features = np.array([feature_values])

if st.button("Predict"): 
    # 将特征分成 "需要标准化的" 和 "不需要标准化的"
    continuous_features = [Gravidity, HCG, Progesterone]
    categorical_features = [
        History_of_pelvic_surgery,
        History_of_cesarean_section,
        Abdominal_tenderness,
        Homogeneous_adnexal_mass,
        Vaginal_bleeding,
    ]
    
    # 只标准化 3 个连续特征
    standardized_continuous = scaler.transform([continuous_features])
    
    # 组合所有特征（标准化后的 + 原始分类特征）
    final_features = np.array([
        standardized_continuous[0][0],  # Gravidity (标准化后)
        standardized_continuous[0][1],  # HCG (标准化后)
        standardized_continuous[0][2],  # Progesterone (标准化后)
        *categorical_features
    ]).reshape(1, -1)
    
    # 预测
    predicted_class = model.predict(final_features)[0]    
    predicted_proba = model.predict_proba(final_features)[0]

    # Display prediction results    
    st.write(f"**Predicted Class:** {predicted_class}(1: Disease, 0: No Disease)")    
    st.write(f"**Prediction Probabilities:** {predicted_proba}")

    # Generate advice based on prediction results    
    probability = predicted_proba[predicted_class] * 100

    if predicted_class == 1:        
        advice = (            
            f"According to our model, you have a high risk of ectopic pregnancy. "            
            f"The model predicts that your probability of having ectopic pregnancy is {probability:.1f}%. "                  
            )    
    else:
        advice = (            
            f"According to our model, you have a low risk of ectopic pregnancy. "            
            f"The model predicts that your probability of not having ectopic pregnancy is {probability:.1f}%. "                   
        )
    st.write(advice)

# SHAP Explanation
    st.subheader("SHAP Force Plot Explanation")
# 获取模型解释器
    explainer_shap = shap.TreeExplainer(model)
# 生成 SHAP 值（使用最终预测的 final_features，即 (1, 8) 维度）
    shap_values = explainer_shap.shap_values(pd.DataFrame(final_features, columns=feature_names))
# 原始数据（未标准化）用于显示标签
    original_feature_values = pd.DataFrame(features, columns=feature_names)
# 显示 SHAP 力力图（根据预测类别）
    if predicted_class == 1:
        shap.force_plot(
        explainer_shap.expected_value[1],
        shap_values[1],  # 预测为 1 类的 SHAP 值
        original_feature_values,  # 显示原始值
        matplotlib=True,
        )
    else:
        shap.force_plot(
        explainer_shap.expected_value[0],
        shap_values[0],  # 预测为 0 类的 SHAP 值
        original_feature_values,
        matplotlib=True,
    )
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=300)
    st.image("shap_force_plot.png", caption='SHAP Force Plot Explanation')