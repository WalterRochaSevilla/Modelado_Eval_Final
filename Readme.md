# Modelado_Eval_Final 🧮

Evaluación final de la materia **Modelado, Dinámica de Sistemas y Simulación (SIS‑224)**  
Universidad Católica Boliviana, CBA | Semestre 1‑2025  
Profesor: VICTOR ADOLFO RODRIGUEZ ESTEVEZ


## 📁 Estructura del repositorio
```python
Modelado_Eval_Final/
├── .ipynb_checkpoints/ # Copias automáticas de Jupyter
├── pycache/ # Archivos compilados de Python
├── Examen Final V1.ipynb # Notebook con la versión completa del examen
├── Examen_pregunta2.ipynb # Solución detallada de la Pregunta 2
├── Examen_pregunta3.ipynb # Solución detallada de la Pregunta 3
├── RedNeuronal.py # Implementación de red neuronal clásica
└── datos_mercurio_seguros.csv # Datos de entrada (mercado/seguros)
```

## 📘 Contenido de los archivos

- **Examen Final V1.ipynb**  
  Notebook principal con un análisis completo de todo el examen: cómo se estructuraron las estrategias, se resolvieron las preguntas, se cargaron datos y se generaron resultados. Ideal para revisar el enfoque general.

- **Examen_pregunta2.ipynb**  
  Enfoque específico para la *Pregunta 2*. Contiene:
  - Carga y limpieza de los datos.
  - Modelado o simulación solicitada.
  - Visualización de resultados y conclusiones.

- **Examen_pregunta3.ipynb**  
  Similar al anterior, pero centrado en la *Pregunta 3*. Incluye todo el pipeline: desde los datos hasta gráficos e inferencias finales.

- **RedNeuronal.py**  
  Script en Python con:
  - Definición e inicialización de arquitectura (capas, funciones de activación).
  - Ruta de entrenamiento y validación.
  - Funciones de predicción o evaluación.
  Útil como módulo usable desde otros notebooks o scripts.

- **datos_mercurio_seguros.csv**  
  Conjunto de datos bruto relacionado al mercado de seguros (o sector Mercurio). Las notebooks extraen, transforman y visualizan variables relevantes para modelado y simulación.

## 🧰 Requisitos e instalación

1. Clona el repositorio:
```bash
git clone https://github.com/Zatt010/NeuroHealth
cd NeuroHealth/predictive_model
```

2. Crea un entorno virtual (opcional):
```bash
python3 -m venv venv
source venv/bin/activate
```
## ▶️ Cómo usar
### Notebooks (.ipynb)
    - Ábrelos con Jupyter:
```bash
jupyter notebook
```
    - Ejecuta cada celda en orden hasta la última para reproducir análisis completos.
### Script (RedNeuronal.py)
    - Ejecuta directamente en terminal/IDE:
```bash
python RedNeuronal.py
```
    - Modifica hiperparámetros (tasa de aprendizaje, epochs, capas) según sea necesario.

## ✅ Resultados esperados

Al correr los notebooks y el script, obtienes:

    - Limpieza y exploración de datos (datos_mercurio_seguros.csv).

    - Modelos de simulación o dinámicas solicitadas para cada pregunta.

    - Entrenamiento y evaluación de una red neuronal.

    - Visualizaciones: gráficos comparativos, curvas de pérdida/precisión, etc.