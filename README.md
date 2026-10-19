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
| Archivo | `data/raw/bank-additional-full.csv` |
| Ficha de variables | `data/raw/bank-additional-names.txt` |
| Instancias | 41.188 |
| Atributos | 20 + salida (`y`) |

**Escenario de predicción (definido):** la predicción se realiza **antes de realizar la llamada**, con el objetivo de decidir a qué clientes contactar (§1). En consecuencia:

- `duration` se **descarta** como característica: no se conoce antes de la llamada y, según la ficha del dataset (`bank-additional-names.txt`), una vez finalizada la llamada el resultado `y` ya es conocido. Solo podría usarse con fines de benchmark, no en el modelo principal.
- El resto del bloque *"last contact of the current campaign"* se evaluó en el EDA (notebook 01, sección 10) con el mismo criterio: **`campaign`, `month`, `day_of_week` y `contact` quedan FUERA** (información generada durante la ejecución de la campaña). **Set final de modelado: 15 features**.

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

> Evidencia reproducible en `notebooks/01_exploratory_analysis.ipynb` (ejecutado con el entorno `.venv`).

- **Dimensiones y tipos**: 41.188 × 21 — 10 numéricas (`int64`/`float64`) y 11 categóricas de texto (incluye `y`). 0 celdas vacías, 0 NaN.
- **Distribución del objetivo**: `no` 36.548 (88,7%) · `yes` 4.640 (11,3%) → desbalance **7,9:1**. Consecuencias: el accuracy no sirve como métrica principal, el split debe estratificarse y el desbalance se evaluará en §6.
- **Categorías**: `job` 12 · `education` 8 · `marital` 4 · `default`/`housing`/`loan`/`poutcome` 3 · `month` 10 (sin ene/feb: ventana de campaña, no es error) · `day_of_week` 5 · `contact` 2.
- **Tratamiento de `unknown`** (missing codificado como categoría, ficha l.72): `default` 20,9% · `education` 4,2% · `housing` y `loan` 2,4% · `job` 0,8% · `marital` 0,2%. **Decisión**: no imputar; mantener `unknown` como categoría con codificación propia (One-Hot) en §6. `default` además tiene solo 3 `yes` (0,0%) → candidata a descartarse.
- **Duplicados**: 12 copias redundantes (24 filas implicadas, 0,03%) → eliminar en §6 (evita doble peso y fuga train/test).
- **Consistencia de centinelas**: `previous=0` ↔ `poutcome=nonexistent` coinciden al 100% (35.563 filas). Pero **4.110 filas tienen `pdays=999` con historia previa** (`previous>0`) → contradicen la ficha; `previous`/`poutcome` son la fuente de verdad histórica y `pdays` deberá tratarse de forma explícita en §6.
- **Distribuciones numéricas**: `age` 17–98 (mediana 38) · `campaign` 1–56 (mediana 2) · `previous` 0–7 · `pdays` dominado por el 999 (Q1=999) · `euribor3m` 0,63–5,04 · `nr.employed` 4.964–5.228. Atípicos IQR: `age` 469 · `campaign` 2.406 · `previous` 5.625 (artefacto estadístico: IQR=0) · `duration` 2.963 → señalizar, no eliminar automáticamente.
- **Relaciones con el objetivo** (tasa base 11,3%): por encima `poutcome=success` **65,1%**, `student` 31,4%, `retired` 25,2%, `contact=cellular` 14,7%; por debajo `blue-collar` 6,9%, `poutcome=nonexistent` 8,8%, `default=unknown` 5,2%. `housing` casi plano (10,8–11,6%). Correlación con `y_bin`: `nr.employed` −0,36 · `euribor3m` −0,31 · `emp.var.rate` −0,30 · `previous` +0,23. `duration` media 220,8 s (`no`) vs 553,2 s (`yes`) → confirma su exclusión por fuga.
- **Multicolinealidad**: los indicadores macro se mueven entre sí (mapa de calor del notebook) → prever regularización en modelos lineales e interpretar importancias con cuidado.
- **Disponibilidad (§3, resuelto)**: 15 features IN (perfil 7 + histórico 3 + macro 5) · 5 FUERA (`duration`, `campaign`, `month`, `day_of_week`, `contact`).

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

Descargar Bank Marketing desde UCI Machine Learning Repository (https://archive.ics.uci.edu/dataset/222/bank+marketing), extraer del zip `bank-additional.zip` los siguientes archivos y colocarlos en:

```text
data/raw/
├── bank-additional-full.csv
└── bank-additional-names.txt
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
