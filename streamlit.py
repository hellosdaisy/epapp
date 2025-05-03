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
    0: 'None',    
    1: 'Less',    
    2: 'Equivalent'
}

HCG_options = {    
    0: 'hCG＜1000',    
    1: '1000≤hCG＜2000',    
    2: '2000≤hCG＜3000',
    3: '3000≤hCG＜4000',    
    4: '4000≤hCG＜5000', 
    5: 'hCG≥5000'
}

# Define feature names
feature_names = [    
    "Gravidity",  "History_of_pelvic_surgery", "History_of_cesarean_section", "Abdominal_tenderness",    
    "Vaginal_bleeding","Homogeneous_adnexal_mass", "HCG", "Progesterone"
]

# Streamlit user interface
st.title("Ectopic Pregnancy Risk Assessment")

# numerical input
Gravidity = st.number_input("Gravidity:", min_value=0, max_value=8, value=1)
Progesterone = st.number_input("Progesterone(ng/ml):", min_value=0.29, max_value=58.0, value=15.0)

#categorical selection
History_of_pelvic_surgery = st.selectbox("History of pelvic surgery :", options=[0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
History_of_cesarean_section = st.selectbox("History of cesarean section :", options=[0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
Abdominal_tenderness= st.selectbox("Abdominal tenderness :", options=[0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
Homogeneous_adnexal_mass = st.selectbox("Homogeneous adnexal mass:", options=[0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
Vaginal_bleeding = st.selectbox("Vaginal bleeding(compare with menstrual flow):", options=list(Vaginal_bleeding_options.keys()), format_func=lambda x: Vaginal_bleeding_options[x])
HCG = st.selectbox("hCG(mIU/ml):", options=list(HCG_options.keys()), format_func=lambda x: HCG_options[x])

# Process inputs and make predictions
feature_values = [Gravidity, History_of_pelvic_surgery, History_of_cesarean_section, Abdominal_tenderness, Vaginal_bleeding, Homogeneous_adnexal_mass, HCG,Progesterone]
features = np.array([feature_values])

# 分离连续变量和分类变量
continuous_features = [Gravidity,Progesterone]
categorical_features=[History_of_pelvic_surgery, History_of_cesarean_section,
       Abdominal_tenderness, Vaginal_bleeding, Homogeneous_adnexal_mass,HCG]

# 对连续变量进行标准化
continuous_features_array = np.array(continuous_features).reshape(1, -1)

# 关键修改：使用 pandas DataFrame 来确保列名
continuous_features_df = pd.DataFrame(continuous_features_array, columns=['Gravidity','Progesterone'])

# 标准化连续变量
continuous_features_standardized = scaler.transform(continuous_features_df)

# 将标准化后的连续变量和原始分类变量合并
# 确保连续特征是二维数组，分类特征是一维数组，合并时要注意维度一致
categorical_features_array = np.array(categorical_features).reshape(1, -1)

# 将标准化后的连续变量和原始分类变量合并
final_features = np.hstack([continuous_features_standardized, categorical_features_array])

# 关键修改：确保 final_features 是一个二维数组，并且用 DataFrame 传递给模型
final_features_df = pd.DataFrame(final_features, columns=feature_names)

if st.button("Predict"): 
    OPTIMAL_THRESHOLD = 0.611
    
    # Predict class and probabilities    
    #predicted_class = model.predict(final_features_df)[0]   
    predicted_proba = model.predict_proba(final_features_df)[0]
    prob_class1 = predicted_proba[1]  # 类别1的概率

    # 根据最优阈值判断类别
    predicted_class = 1 if prob_class1 >= OPTIMAL_THRESHOLD else 0


    # Display prediction results       
    st.write(f"**Predicted Probability:** {prob_class1:.1%}")
    st.write(f"**Decision Threshold:** {OPTIMAL_THRESHOLD:.0%} (optimized for clinical utility)")
    st.write(f"**Predicted Class:** {predicted_class}(1: High risk of EP, 0: Low risk of EP)") 

    # Generate advice based on prediction results    
    #probability = predicted_proba[predicted_class] * 100

    #if predicted_class == 1:        
    #    advice = (            
    #        f"According to our model, you have a high risk of ectopic pregnancy. "            
    #        f"The model predicts that your probability of having ectopic pregnancy is {probability:.1f}%. "           )    
    #else:
    #    advice = (            
    #        f"According to our model, you have a low risk of ectopic pregnancy. "            
    #        f"The model predicts that your probability of not having ectopic pregnancy is {probability:.1f}%. "                   
    #    )
    #st.write(advice)

# SHAP Explanation    
    st.subheader("SHAP Force Plot Explanation")    
    explainer_shap = shap.TreeExplainer(model)    
    shap_values = explainer_shap.shap_values(final_features_df)    
    # 将标准化前的原始数据存储在变量中
    original_feature_values = pd.DataFrame(features, columns=feature_names)

    # Display the SHAP force plot for the predicted class    
    if predicted_class == 1:        
        shap.force_plot(explainer_shap.expected_value[1], shap_values[1], original_feature_values, matplotlib=True)    
    else:        
        shap.force_plot(explainer_shap.expected_value[0], shap_values[0], original_feature_values, matplotlib=True)    
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)    
    st.image("shap_force_plot.png", caption='SHAP Force Plot Explanation')