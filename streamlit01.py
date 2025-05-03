import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# 加载模型和标准化器
model = joblib.load('catboost.pkl')
scaler = joblib.load('scaler.pkl') 

# 定义特征选项和名称(保持不变)
Vaginal_bleeding_options = {0: 'None', 1: 'Less', 2: 'Equivalent'}
HCG_options = {
    0: 'hCG＜1000', 1: '1000≤hCG＜2000', 2: '2000≤hCG＜3000',
    3: '3000≤hCG＜4000', 4: '4000≤hCG＜5000', 5: 'hCG≥5000'
}
feature_names = [
    "Gravidity", "History_of_pelvic_surgery", "History_of_cesarean_section",
    "Abdominal_tenderness", "Vaginal_bleeding", "Homogeneous_adnexal_mass",
    "HCG", "Progesterone"
]

# 界面布局
st.title("Ectopic Pregnancy Risk Assessment")

# 用户输入(保持不变)
Gravidity = st.number_input("Gravidity:", min_value=0, max_value=8, value=1)
Progesterone = st.number_input("Progesterone(ng/ml):", min_value=0.29, max_value=58.0, value=15.0)
History_of_pelvic_surgery = st.selectbox("History of pelvic surgery:", options=[0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
History_of_cesarean_section = st.selectbox("History of cesarean section:", options=[0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
Abdominal_tenderness = st.selectbox("Abdominal tenderness:", options=[0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
Homogeneous_adnexal_mass = st.selectbox("Homogeneous adnexal mass:", options=[0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
Vaginal_bleeding = st.selectbox("Vaginal bleeding (compare with menstrual flow):", 
                              options=list(Vaginal_bleeding_options.keys()), 
                              format_func=lambda x: Vaginal_bleeding_options[x])
HCG = st.selectbox("hCG(mIU/ml):", options=list(HCG_options.keys()), format_func=lambda x: HCG_options[x])

if st.button("Predict"):
    # 1. 收集所有特征值
    feature_values = {
        "Gravidity": Gravidity,
        "History_of_pelvic_surgery": History_of_pelvic_surgery,
        "History_of_cesarean_section": History_of_cesarean_section,
        "Abdominal_tenderness": Abdominal_tenderness,
        "Vaginal_bleeding": Vaginal_bleeding,
        "Homogeneous_adnexal_mass": Homogeneous_adnexal_mass,
        "HCG": HCG,
        "Progesterone": Progesterone
    }
    
    # 2. 创建DataFrame
    input_df = pd.DataFrame([feature_values])
    
    # 3. 仅对连续变量进行标准化(关键修复点)
    continuous_cols = ['Gravidity', 'Progesterone']
    input_df[continuous_cols] = scaler.transform(input_df[continuous_cols])
    
    # 4. 预测
    OPTIMAL_THRESHOLD = 0.611
    predicted_proba = model.predict_proba(input_df)[0]
    prob_class1 = predicted_proba[1]
    predicted_class = 1 if prob_class1 >= OPTIMAL_THRESHOLD else 0
    
    # 5. 显示结果
    st.write(f"**Predicted Probability:** {prob_class1:.1%}")
    st.write(f"**Predicted Class:** {predicted_class} (1: High risk of EP, 0: Low risk of EP)")
    
    # 6. SHAP解释
    st.subheader("Model Explanation")
    explainer = shap.TreeExplainer(model)
    
    # 获取SHAP值(处理二分类情况)
    shap_values = explainer.shap_values(input_df)
    if isinstance(shap_values, list):  # 二分类模型
        shap_values = shap_values[1]  # 使用正类的SHAP值
    
    # 创建force plot
    plt.figure()
    shap.force_plot(
        explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value,
        shap_values[0],
        input_df.iloc[0],
        feature_names=feature_names,
        matplotlib=True,
        show=False
    )
    st.pyplot(plt.gcf())
    plt.close()
