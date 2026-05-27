"""
Sistema de Predicción de Riesgo de Diabetes
XGBoost + SHAP — Aplicación Streamlit
Autor: Bryan Anthony López Guerrero
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import shap
import pickle
import os

# ── Configuración de página ───────────────────────────────
st.set_page_config(
    page_title="Predictor de Riesgo de Diabetes",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Estilos personalizados ────────────────────────────────
st.markdown("""
<style>
.main-title {
    font-size: 2.2rem;
    font-weight: 800;
    color: #1D3557;
    margin-bottom: 0.2rem;
}
.subtitle {
    font-size: 1rem;
    color: #6C757D;
    margin-bottom: 1.5rem;
}
.metric-card {
    background: white;
    border: 1px solid #DEE2E6;
    border-radius: 8px;
    padding: 1rem;
    text-align: center;
}
.risk-high {
    background: #F8D7DA;
    border: 2px solid #E63946;
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
}
.risk-low {
    background: #D4EDDA;
    border: 2px solid #2D6A4F;
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
}
.risk-medium {
    background: #FFF3CD;
    border: 2px solid #F4A261;
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
}
.info-box {
    background: #EBF5FB;
    border-left: 4px solid #457B9D;
    border-radius: 4px;
    padding: 1rem;
    margin: 0.5rem 0;
    color: #1D3557;
}
</style>
""", unsafe_allow_html=True)


# ── Cargar modelo ─────────────────────────────────────────
@st.cache_resource
def cargar_modelo():
    model_path = os.path.join(os.path.dirname(__file__),
                              '../models/xgboost_diabetes.pkl')
    with open(model_path, 'rb') as f:
        bundle = pickle.load(f)
    return bundle['model'], bundle['scaler'], bundle['features']


try:
    model, scaler, features = cargar_modelo()
    modelo_cargado = True
except Exception as e:
    modelo_cargado = False
    st.error(f"Error cargando el modelo: {e}")


# ── Sidebar — Inputs del paciente ─────────────────────────
with st.sidebar:
    st.markdown("## 🩺 Datos del Paciente")
    st.markdown("Ingresa los indicadores clínicos para obtener la predicción.")
    st.markdown("---")

    pregnancies = st.slider(
        "Número de embarazos",
        min_value=0, max_value=17, value=3,
        help="Número de veces que ha estado embarazada")

    glucose = st.slider(
        "Glucosa en plasma (mg/dL)",
        min_value=44, max_value=199, value=120,
        help="Concentración de glucosa en plasma a las 2 horas en una prueba oral de tolerancia a la glucosa")

    blood_pressure = st.slider(
        "Presión arterial diastólica (mm Hg)",
        min_value=24, max_value=122, value=70,
        help="Presión arterial diastólica en mm Hg")

    skin_thickness = st.slider(
        "Grosor del pliegue cutáneo (mm)",
        min_value=0, max_value=99, value=20,
        help="Grosor del pliegue cutáneo del tríceps en mm")

    insulin = st.slider(
        "Insulina sérica a 2h (mu U/ml)",
        min_value=0, max_value=846, value=80,
        help="Insulina sérica a las 2 horas en mu U/ml")

    bmi = st.slider(
        "BMI — Índice de masa corporal",
        min_value=0.0, max_value=67.0, value=32.0, step=0.1,
        help="Peso en kg / (altura en m)²")

    dpf = st.slider(
        "Función de pedigrí de diabetes",
        min_value=0.078, max_value=2.420,
        value=0.470, step=0.001,
        help="Función que puntúa la probabilidad de diabetes según historial familiar")

    age = st.slider(
        "Edad (años)",
        min_value=21, max_value=81, value=33,
        help="Edad en años")

    st.markdown("---")
    predecir = st.button("🔍 Analizar riesgo", use_container_width=True,
                         type="primary")
    st.markdown("---")
    st.markdown("""
    <div class='info-box'>
    <b>⚠️ Aviso importante</b><br>
    Esta herramienta es únicamente para fines educativos y de demostración.
    No reemplaza el diagnóstico médico profesional.
    </div>
    """, unsafe_allow_html=True)


# ── Página principal ──────────────────────────────────────
st.markdown("<div class='main-title'>🩺 Sistema de Predicción de Riesgo de Diabetes</div>",
            unsafe_allow_html=True)
st.markdown("<div class='subtitle'>XGBoost + Explicabilidad SHAP · Dataset Pima Indians Diabetes · "
            "ROC-AUC = 0.89 | Autor: Bryan Anthony López Guerrero</div>",
            unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "🎯 Predicción individual",
    "📊 Métricas del modelo",
    "ℹ️ Sobre el proyecto"
])


# ══════════════════════════════════════════════════════════
# TAB 1 — PREDICCIÓN INDIVIDUAL
# ══════════════════════════════════════════════════════════
with tab1:
    if not predecir:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("### 📋 Resumen de los datos ingresados")
            datos_tabla = pd.DataFrame({
                'Variable': ['Embarazos', 'Glucosa (mg/dL)',
                             'Presión arterial (mm Hg)',
                             'Grosor piel (mm)', 'Insulina (mu U/ml)',
                             'BMI', 'Función pedigrí', 'Edad'],
                'Valor': [pregnancies, glucose, blood_pressure,
                          skin_thickness, insulin, bmi, dpf, age],
                'Rango normal': ['0–5', '70–100', '60–80',
                                 '10–30', '16–166', '18.5–24.9',
                                 '0.1–0.5', '—']
            })
            st.dataframe(datos_tabla, use_container_width=True,
                         hide_index=True)
        with col2:
            st.markdown("### 🚀 Cómo usar esta herramienta")
            st.markdown("""
            1. **Ajusta los sliders** del panel izquierdo con los datos clínicos del paciente
            2. **Haz clic en "Analizar riesgo"** para obtener la predicción
            3. **Revisa la explicación SHAP** para entender qué variables influyeron más
            4. **Interpreta el resultado** en el contexto clínico apropiado

            El modelo fue entrenado con **768 pacientes** del dataset Pima Indians Diabetes
            (UCI Machine Learning Repository) y validado con un **ROC-AUC de 0.89**.
            """)
            st.info("👈 Ajusta los valores en el panel izquierdo y haz clic en **Analizar riesgo**")

    else:
        if not modelo_cargado:
            st.error("El modelo no está disponible.")
        else:
            # Preparar input
            input_data = np.array([[pregnancies, glucose, blood_pressure,
                                    skin_thickness, insulin, bmi, dpf, age]])
            input_scaled = scaler.transform(input_data)
            input_df = pd.DataFrame(input_scaled, columns=features)

            # Predicción
            prob = model.predict_proba(input_scaled)[0][1]
            pred = int(prob >= 0.5)

            # Categorizar riesgo
            if prob >= 0.65:
                riesgo = "ALTO"
                css_class = "risk-high"
                emoji = "🔴"
                color_riesgo = "#E63946"
            elif prob >= 0.35:
                riesgo = "MODERADO"
                css_class = "risk-medium"
                emoji = "🟡"
                color_riesgo = "#F4A261"
            else:
                riesgo = "BAJO"
                css_class = "risk-low"
                emoji = "🟢"
                color_riesgo = "#2D6A4F"

            # ── Resultado principal ────────────────────────
            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                st.markdown(f"""
                <div class='{css_class}'>
                <h2 style='color:{color_riesgo};margin:0'>{emoji} Riesgo {riesgo}</h2>
                <h1 style='color:{color_riesgo};margin:0.5rem 0'>{prob:.1%}</h1>
                <p style='margin:0;color:#6C757D'>Probabilidad de diabetes</p>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.metric("Diagnóstico predicho",
                          "Con diabetes" if pred == 1 else "Sin diabetes",
                          delta=f"Umbral: 50%")
                st.metric("Glucosa ingresada", f"{glucose} mg/dL",
                          delta=f"{'↑ Elevada' if glucose > 140 else '✓ Normal' if glucose >= 70 else '↓ Baja'}")

            with col3:
                st.metric("BMI ingresado", f"{bmi:.1f}",
                          delta=f"{'↑ Obesidad' if bmi >= 30 else '✓ Normal' if bmi >= 18.5 else '↓ Bajo peso'}")
                st.metric("Edad", f"{age} años",
                          delta=f"{'Factor de riesgo' if age >= 45 else 'Menor riesgo por edad'}")

            st.markdown("---")

            # ── SHAP Waterfall ─────────────────────────────
            st.markdown("### 🔍 Explicación SHAP — ¿Por qué el modelo tomó esta decisión?")

            explainer   = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(input_df)
            shap_ind    = shap_values[0]

            nombres_legibles = {
                'Pregnancies': 'Embarazos',
                'Glucose': 'Glucosa',
                'BloodPressure': 'Presión arterial',
                'SkinThickness': 'Grosor piel',
                'Insulin': 'Insulina',
                'BMI': 'BMI',
                'DiabetesPedigreeFunction': 'Pedigrí diabetes',
                'Age': 'Edad'
            }
            valores_originales = dict(zip(features,
                [pregnancies, glucose, blood_pressure,
                 skin_thickness, insulin, bmi, dpf, age]))

            idx_s   = np.argsort(np.abs(shap_ind))
            feat_s  = [features[i] for i in idx_s]
            shap_s  = shap_ind[idx_s]
            labels_s = [f"{nombres_legibles[f]}\n({valores_originales[f]:.1f})"
                        for f in feat_s]
            cols_w  = ['#E63946' if v > 0 else '#2D6A4F' for v in shap_s]

            col_shap1, col_shap2 = st.columns([3, 2])
            with col_shap1:
                fig, ax = plt.subplots(figsize=(9, 5))
                bars = ax.barh(labels_s, shap_s, color=cols_w,
                               alpha=0.85, height=0.6)
                ax.axvline(0, color='black', lw=1.5)
                for bar, val in zip(bars, shap_s):
                    offset = 0.003 if val >= 0 else -0.003
                    ha = 'left' if val >= 0 else 'right'
                    ax.text(val + offset,
                            bar.get_y() + bar.get_height()/2,
                            f'{val:+.3f}',
                            va='center', ha=ha, fontsize=9)
                patch_pos = mpatches.Patch(color='#E63946',
                                           label='↑ Aumenta riesgo')
                patch_neg = mpatches.Patch(color='#2D6A4F',
                                           label='↓ Reduce riesgo')
                ax.legend(handles=[patch_pos, patch_neg], fontsize=9,
                          loc='lower right')
                ax.set_xlabel('Valor SHAP (impacto en la predicción)')
                ax.set_title(f'Explicación SHAP — Probabilidad predicha: {prob:.1%}',
                             fontsize=11, fontweight='bold')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.grid(axis='x', alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

            with col_shap2:
                st.markdown("#### 📖 Cómo leer este gráfico")
                st.markdown("""
                Cada barra muestra cómo esa variable **empuja la predicción**
                hacia o contra el riesgo de diabetes:

                🔴 **Barras rojas** → aumentan el riesgo predicho

                🟢 **Barras verdes** → reducen el riesgo predicho

                El valor numérico indica la **magnitud del impacto**.
                La variable con la barra más larga es la más influyente
                en esta predicción específica.
                """)

                # Variable más influyente
                top_idx = np.argmax(np.abs(shap_ind))
                top_feat = features[top_idx]
                top_val  = shap_ind[top_idx]
                direccion = "aumenta" if top_val > 0 else "reduce"
                st.markdown(f"""
                <div class='info-box'>
                <b>Variable más influyente:</b><br>
                <b>{nombres_legibles[top_feat]}</b> {direccion} el riesgo
                en {abs(top_val):.3f} puntos SHAP para este paciente.
                </div>
                """, unsafe_allow_html=True)

            # ── Tabla de contribuciones ────────────────────
            st.markdown("#### 📋 Contribución de cada variable")
            tabla_shap = pd.DataFrame({
                'Variable': [nombres_legibles[f] for f in features],
                'Valor ingresado': [pregnancies, glucose, blood_pressure,
                                    skin_thickness, insulin, bmi, dpf, age],
                'Valor SHAP': [round(v, 4) for v in shap_ind],
                'Efecto': ['↑ Aumenta riesgo' if v > 0 else '↓ Reduce riesgo'
                           for v in shap_ind],
            }).sort_values('Valor SHAP', key=abs, ascending=False)

            st.dataframe(
                tabla_shap.style.background_gradient(
                    subset=['Valor SHAP'],
                    cmap='RdYlGn_r',
                    vmin=-0.5, vmax=0.5),
                use_container_width=True,
                hide_index=True
            )


# ══════════════════════════════════════════════════════════
# TAB 2 — MÉTRICAS DEL MODELO
# ══════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 📊 Rendimiento del modelo XGBoost")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("ROC-AUC", "0.8900", "↑ vs LR: +0.0274")
    col2.metric("F1-Score", "0.7475", "Clase positiva")
    col3.metric("CV ROC-AUC", "0.9383 ± 0.007", "5-fold estratificado")
    col4.metric("Precisión global", "84.4%", "Test set (n=154)")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Comparación de modelos")
        df_comp = pd.DataFrame({
            'Modelo': ['Logistic Regression', 'Random Forest', 'XGBoost ★'],
            'ROC-AUC': [0.8626, 0.8601, 0.8900],
            'Seleccionado': ['No', 'No', '✓ Sí']
        })
        st.dataframe(df_comp, use_container_width=True, hide_index=True)

    with col2:
        st.markdown("#### Reporte de clasificación")
        df_rep = pd.DataFrame({
            'Clase': ['Sin diabetes', 'Con diabetes'],
            'Precision': [0.84, 0.82],
            'Recall': [0.92, 0.69],
            'F1-Score': [0.88, 0.75],
            'Support': [100, 54]
        })
        st.dataframe(df_rep, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("#### 🔬 Configuración del modelo")
    st.code("""
XGBClassifier(
    n_estimators        = 300,
    max_depth           = 4,
    learning_rate       = 0.05,
    subsample           = 0.8,
    colsample_bytree    = 0.8,
    reg_alpha           = 0.1,
    reg_lambda          = 1.0,
    min_child_weight    = 3,
    gamma               = 0.1,
    eval_metric         = 'logloss',
    random_state        = 42
)

# Balanceo de clases: SMOTE (400 vs 214 → 400 vs 400)
# Validación: StratifiedKFold (5 folds)
# Preprocesamiento: StandardScaler + imputación por mediana por clase
    """, language='python')


# ══════════════════════════════════════════════════════════
# TAB 3 — SOBRE EL PROYECTO
# ══════════════════════════════════════════════════════════
with tab3:
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("### 📖 Sobre este proyecto")
        st.markdown("""
        Este proyecto forma parte del portafolio de Data Science de
        **Bryan Anthony López Guerrero**, Ingeniero en TI y Máster en
        Visual Analytics y Big Data.

        **Objetivo:** Desarrollar un sistema de predicción de riesgo de
        diabetes que combine precisión técnica (XGBoost) con explicabilidad
        clínica (SHAP) y usabilidad (Streamlit).

        **Dataset:** Pima Indians Diabetes — UCI Machine Learning Repository.
        768 pacientes, 8 variables clínicas, variable objetivo binaria.

        **Pipeline técnico:**
        1. Imputación de ceros imposibles con mediana por clase
        2. Split estratificado 80/20
        3. Estandarización con StandardScaler
        4. Balanceo con SMOTE
        5. Entrenamiento XGBoost con hiperparámetros optimizados
        6. Explicabilidad con SHAP (global e individual)
        7. Despliegue con Streamlit
        """)

    with col2:
        st.markdown("### 👤 Autor")
        st.markdown("""
        **Bryan Anthony López Guerrero**

        📧 anthonyxm15@gmail.com

        🎓 Ing. Tecnologías de la Información

        🎓 Máster en Visual Analytics y Big Data

        📜 Especialista en Big Data e IA

        ---

        **Stack de este proyecto:**
        - Python 3.10
        - XGBoost 2.0
        - SHAP 0.44
        - scikit-learn 1.3
        - imbalanced-learn
        - Streamlit
        - Power BI
        """)

    st.markdown("---")
    st.markdown("""
    <div style='text-align:center;color:#6C757D;font-size:0.85rem'>
    ⚠️ Esta aplicación es únicamente para fines educativos y de demostración de competencias técnicas.
    No constituye ni reemplaza un diagnóstico médico profesional.
    Ante cualquier duda clínica, consulte siempre a un profesional de la salud.
    </div>
    """, unsafe_allow_html=True)
