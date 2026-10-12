# Bank Marketing Classification

> **Guía de README · Proyecto de clasificación supervisada**
> Documento vivo: completar cada sección únicamente cuando exista evidencia en el análisis.

## 1. Problema

Las campañas de marketing telefónico requieren tiempo y recursos. Contactar indiscriminadamente a todos los clientes puede resultar ineficiente.

Este proyecto estudia si los datos históricos permiten identificar clientes con mayor probabilidad de contratar un depósito a plazo, sin utilizar información que no estaría disponible en el momento de realizar la predicción.

## 2. Objetivo

Construir y comparar modelos de clasificación supervisada para predecir la suscripción a un depósito a plazo, mediante un proceso reproducible que incluya análisis exploratorio, preprocesamiento, baseline, validación, comparación e interpretación de resultados.

## 3. Dataset

| Campo | Valor |
| --- | --- |
| Nombre | Bank Marketing |
| Fuente | UCI Machine Learning Repository |
| Área | Negocios |
| Tarea | Clasificación binaria |
| Variable objetivo | `y` (yes/no) |
| Licencia | Creative Commons Attribution 4.0 |
| DOI | 10.24432/C5K306 |
| Archivo previsto | `bank-additional-full.csv` |

**Consideración sobre `duration`:** la variable `duration` representa la duración del último contacto. Su tratamiento debe decidirse después de definir exactamente en qué momento se realizaría la predicción, para evitar utilizar información que no estaría disponible en ese escenario.

## 4. Preguntas de análisis

- ¿Cómo se distribuye la variable objetivo?
- ¿Existe un desbalance significativo entre las clases?
- ¿Qué variables son numéricas, categóricas, binarias o temporales?
- ¿Cómo deben interpretarse las categorías `unknown`?
- ¿Existen registros duplicados o valores inconsistentes?
- ¿Qué características muestran relaciones relevantes con el objetivo?
- ¿Qué información estaría disponible en el momento real de la predicción?
- ¿Qué variables podrían introducir fuga de información?
- ¿Qué métrica debería priorizarse según el objetivo del proyecto?

## 5. Análisis Exploratorio de Datos (EDA)

Se completará después de examinar el dataset. Documentar:

- Dimensiones y tipos de variables.
- Distribución del objetivo.
- Categorías y valores únicos.
- Tratamiento conceptual de `unknown`.
- Duplicados y consistencia.
- Distribuciones numéricas y posibles valores atípicos.
- Relaciones entre características y con el objetivo.
- Principales hallazgos.

## 6. Preprocesamiento

Las decisiones se establecerán a partir del EDA. Documentar:

- Separación entre características y objetivo.
- División de entrenamiento y prueba.
- Estratificación, si corresponde.
- Tratamiento de variables numéricas y categóricas.
- Codificación y escalado cuando el modelo lo requiera.
- Tratamiento del desbalance, si resulta necesario.
- Uso de `Pipeline` y `ColumnTransformer` cuando corresponda.
- Medidas utilizadas para evitar data leakage.

## 7. Baseline

- Estrategia o modelo utilizado.
- Métricas obtenidas.
- Criterio de comparación.
- Limitaciones del baseline.

## 8. Modelos evaluados

Elegir los modelos después del EDA y del baseline. Para cada uno documentar:

- Motivo de selección.
- Preprocesamiento requerido.
- Hiperparámetros relevantes.
- Ventajas y limitaciones.
- Coste computacional cuando resulte pertinente.

## 9. Estrategia de validación

- División de entrenamiento y prueba.
- Estratificación, si corresponde.
- Estrategia de validación cruzada.
- Prevención de leakage.
- Métricas evaluadas.
- Métrica principal y su justificación.

## 10. Resultados

Completar exclusivamente con resultados reproducibles. Tabla sugerida:

| Modelo | Accuracy | Precision | Recall | F1-score | ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| Baseline | Pendiente | Pendiente | Pendiente | Pendiente | Pendiente |

- Matriz de confusión.
- Curvas de evaluación cuando correspondan.
- Variabilidad durante la validación cruzada.
- Tiempos de entrenamiento/predicción si aportan información útil.
- Interpretación de errores y comparación entre modelos.

## 11. Modelo seleccionado

La selección no se basará únicamente en la métrica más alta. Considerar:

- Métrica principal y estabilidad.
- Falsos positivos y falsos negativos.
- Interpretabilidad y complejidad.
- Uso previsto y disponibilidad real de variables.
- Principales trade-offs.

## 12. Limitaciones

- Antigüedad y contexto geográfico de los datos.
- Representatividad del dataset.
- Desbalance entre clases.
- Categorías ambiguas o desconocidas.
- Variables no disponibles en determinados escenarios.
- Posibles fuentes de sesgo.
- Límites de generalización.
- Afirmaciones que los resultados no permiten realizar.

## 13. Conclusiones

- Qué se descubrió durante el análisis.
- Qué modelo presentó el comportamiento más adecuado.
- Cuál fue el criterio de selección.
- Qué errores fueron más relevantes.
- Qué aprendimos del experimento.
- Qué utilidad práctica y limitaciones tendría la solución.

## 14. Tecnologías

Lista inicial, a conservar solo si realmente se utiliza:

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook / JupyterLab
- Git y GitHub

## 15. Estructura del repositorio

```text
Bank-Marketing-Prediction/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   └── 01_exploratory_analysis.ipynb
├── src/
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE
```

Los datos originales deben conservarse sin modificaciones en `data/raw/`. Los datos transformados, si fuera necesario almacenarlos, irían en `data/processed/`.

## 16. Reproducibilidad

Objetivo: permitir que otra persona parta de un clon limpio y pueda ejecutar el proyecto sin adivinar pasos.

### Requisitos

- Python 3.x
- Git
- Jupyter Notebook o JupyterLab

### 1. Clonar el repositorio

```bash
git clone https://github.com/IvanRavarotto/Bank-Marketing-Prediction.git
cd Bank-Marketing-Prediction
```

### 2. Crear un entorno virtual

```bash
python -m venv .venv
```

### 3. Activar el entorno

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Linux / macOS:

```bash
source .venv/bin/activate
```

### 4. Instalar dependencias

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Obtener el dataset

Descargar Bank Marketing desde UCI Machine Learning Repository, extraer `bank-additional-full.csv` y colocarlo en:

```text
data/raw/bank-additional-full.csv
```

El archivo original debe conservarse sin modificaciones.

### 6. Iniciar Jupyter

```bash
jupyter lab
```

Abrir:

```text
notebooks/01_exploratory_analysis.ipynb
```

### 7. Reproducir el análisis

Los notebooks estarán numerados según el orden previsto de ejecución. Ejecutarlos en orden y usando un kernel asociado al entorno virtual `.venv`.

> **Nota:** esta sección deberá actualizarse conforme evolucione la implementación para reflejar el procedimiento real de reproducción.

## 17. Próximos pasos

- Completar la comprensión y documentación del dataset.
- Realizar el análisis exploratorio.
- Definir el escenario exacto de predicción.
- Establecer el baseline.
- Diseñar el pipeline de preprocesamiento.
- Evaluar modelos bajo una estrategia común de validación.
- Documentar resultados, limitaciones y conclusiones.

## 18. Referencias

- UCI Machine Learning Repository, Bank Marketing.
- Moro, S., Cortez, P. y Rita, P. A Data-Driven Approach to Predict the Success of Bank Telemarketing.
- Documentación oficial de las herramientas utilizadas.

---

### Reglas de uso de esta guía

- No inventar resultados, modelos o decisiones metodológicas.
- Completar cada sección a medida que avance el proyecto.
- Toda cifra final del README debe poder reproducirse.
- Documentar por qué se tomó una decisión, no solo qué se hizo.
