import streamlit as st
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ======================
# ESTILOS PERSONALIZADOS
# ======================
st.markdown("""
<style>
    body {
        background-color: #f0f2f6;
    }
    h1 {
        color: #2c3e50;
        text-align: center;
    }
    .stButton>button {
        background-color: #3498db;
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 100%;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ======================
# TÍTULO Y DESCRIPCIÓN
# ======================
st.title("Método de Newton-Raphson 🧮")
st.write("""
Este programa implementa el **Método de Newton-Raphson** para encontrar raíces de funciones.
👉 Ingresa la función, el valor inicial, la tolerancia y el número máximo de iteraciones.
""")

# ======================
# ENTRADAS DE USUARIO
# ======================
funcion_str = st.text_input("Ingresa la función en x:", "1-x^2-atan(x)")
x0 = st.number_input("Valor inicial x0:", value=0.5)
tol = st.number_input("Tolerancia:", value=1e-6, format="%.1e")
max_iter = st.number_input("Número máximo de iteraciones:", value=10, step=1)

# ======================
# BOTÓN PARA CALCULAR
# ======================
if st.button("Calcular raíz"):
    x = sp.symbols('x')
    f = sp.sympify(funcion_str)
    f_prime = sp.diff(f, x)

    f_lamb = sp.lambdify(x, f, 'numpy')
    f_prime_lamb = sp.lambdify(x, f_prime, 'numpy')

    xn = x0
    historial = []
    for i in range(int(max_iter)):
        fxn = f_lamb(xn)
        fpxn = f_prime_lamb(xn)
        if fpxn == 0:
            st.error("⚠️ Derivada nula, no se puede continuar.")
            break
        x_next = xn - fxn/fpxn
        historial.append((i+1, x_next))
        if abs(x_next - xn) < tol:
            break
        xn = x_next

    if historial:
        # ======================
        # MOSTRAR TABLA DE ITERACIONES
        # ======================
        df = pd.DataFrame(historial, columns=["Iteración", "x"])
        st.write("### 📋 Iteraciones")
        st.table(df)

        # ======================
        # MOSTRAR RAÍZ FINAL
        # ======================
        raiz = historial[-1][1]
        st.success(f"✅ Raíz aproximada: {raiz:.6f}")

        # ======================
        # GRÁFICA DE LA FUNCIÓN
        # ======================
        x_vals = np.linspace(raiz-2, raiz+2, 400)
        y_vals = f_lamb(x_vals)

        fig, ax = plt.subplots()
        ax.axhline(0, color='black', linewidth=0.8)
        ax.plot(x_vals, y_vals, label=f"f(x)={funcion_str}")
        ax.plot(raiz, f_lamb(raiz), 'ro', label=f"Raíz ≈ {raiz:.6f}")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
