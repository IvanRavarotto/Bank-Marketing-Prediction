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
- El resto del bloque *"last contact of the current campaign"* se evaluó en el EDA (notebook 01, sección 10) con el mismo criterio: **`campaign`, `month`, `day_of_week` y `contact` quedan FUERA** (información generada durante la ejecución de la campaña).
- Durante el preprocesamiento (§6) se descartaron además **`pdays`** (el 96,3% de sus valores es el centinela `999`, no fiable) y **`default`** (solo 3 `yes` en 41.188 filas). **Set final de modelado: 13 features** (7 numéricas + 6 categóricas).

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

> Evidencia reproducible en `notebooks/02_preprocessing.ipynb` y código en `src/preprocessing.py`.

- **Separación características/objetivo**: `X` con **13 features** (7 numéricas + 6 categóricas) e `y` binaria (`yes=1`, `no=0`).
- **Duplicados**: se eliminaron las **12 filas duplicadas exactas** antes del split → 41.176 filas (41.188 − 12).
- **Variables excluidas**: disponibilidad (§3): `duration`, `campaign`, `month`, `day_of_week`, `contact`; centinela no fiable: `pdays` (39.673/41.188 = 96,3% son `999`, 4.110 filas con historia previa lo contradicen); baja información: `default` (3 `yes`, 0,007%; 20,9% `unknown`).
- **`contactado_antes` derivado**: descartado por **redundancia exacta** con `poutcome` (`previous=0` ↔ `poutcome=nonexistent` coinciden al 100%, verificado en EDA).
- **División train/test**: 80/20 → **32.940 train / 8.236 test**, con `random_state=42`.
- **Estratificación**: sí, por `y`; la proporción se mantiene en **88,73% `no` / 11,27% `yes`** en train y test.
- **Numéricas**: `StandardScaler` (media 0, desvío 1). **Categóricas**: `OneHotEncoder(handle_unknown='ignore')`. `unknown` se conserva como categoría propia (no se imputa).
- **Atípicos**: no se eliminan (el EDA mostró que no son errores); los modelos lineales se apoyan en regularización.
- **`Pipeline` + `ColumnTransformer`**: el preprocesador se ajusta **solo con train** y viaja dentro del pipeline del modelo (re-ajuste por fold en validación cruzada). El `ColumnTransformer` produce **40 columnas** a partir de las 13 features.
- **Data leakage evitado**: split antes del ajuste; escalador ajustado solo con train (medias difieren de las del dataset completo, verificado); sin variables no disponibles; sin codificación basada en el objetivo; duplicados eliminados antes del split.
- **Desbalance**: decisión no asumida; se evaluó empíricamente en §7.

## 7. Baseline

> Evidencia reproducible en `notebooks/03_baseline.ipynb` (split holdout 80/20, `random_state=42`).

- **Estrategia**: dos niveles. (0) referencia trivial `DummyClassifier(strategy='prior')` = predecir siempre `no`; (1) **regresión logística** (L2, `C=1`, `max_iter=1000`) con el pipeline de §6, modelo simple, interpretable y probabilístico.
- **Métricas obtenidas** (clase positiva `yes`, test 8.236 filas, tasa base 11,27%):

| Modelo | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
| --- | --- | --- | --- | --- | --- | --- |
| Dummy (prior) | 0,887 | 0,000 | 0,000 | 0,000 | 0,500 | 0,113 |
| Regresión logística | 0,898 | 0,693 | 0,168 | 0,271 | 0,776 | 0,430 |
| Regresión logística (balanceada) | 0,773 | 0,290 | 0,699 | 0,410 | 0,777 | 0,429 |

- **Criterio de comparación**: se prioriza **PR-AUC** y **recall/F1 de `yes`** por el desbalance (7,9:1); el *accuracy* **no decide** (el dummy ya alcanza 88,7% sin aprender). El modelo sube PR-AUC de 0,113 → 0,430 y ROC-AUC de 0,500 → 0,776.
- **Desbalance**: `class_weight='balanced'` sube el recall de 0,168 → **0,699** (−40,3 pp de precision, ROC-AUC casi igual), pero con umbral 0,5 genera **1.589 falsos positivos** (19% de los `no`). Uso previsto: ranking de clientes (scores), no decisión con umbral fijo.
- **Interpretación descriptiva** (coeficientes de mayor magnitud; no causal): `poutcome_failure` −1,08 · `emp.var.rate` −0,79 · `poutcome_success` +0,78 · `poutcome_nonexistent` −0,52 · `nr.employed` −0,37.
- **Limitaciones del baseline**: una sola partición train/test, sin ajuste de hiperparámetros, umbral fijo en 0,5 y un modelo lineal → comparación más robusta en §8–§9.

## 8. Modelos evaluados

> Evidencia reproducible en `notebooks/04_models.ipynb` (comparación exploratoria) y `notebooks/05_validation.ipynb` (§9). Candidatos declarados en `src/models.py`.

| Modelo | Motivo de selección | Preprocesamiento | Ventajas | Limitaciones | t entrenamiento (holdout) |
| --- | --- | --- | --- | --- | --- |
| Regresión logística | Baseline lineal interpretable | Escalado + One-Hot | Coeficientes legibles, rápido | Solo relaciones lineales | ~0,4 s |
| Regresión logística balanceada | Variante con `class_weight='balanced'` | Ídem | Mejor recall | Baja precision | ~0,5 s |
| Árbol de decisión | Reglas no lineales simples | One-Hot | Interpretable | Inestable | ~0,4 s |
| Random Forest | Ensamble robusto | One-Hot | No linealidad; importancias | Caja seminegra; ~8 s | ~8 s |
| Gradient Boosting | Ensamble secuencial de alto desempeño | One-Hot | Mejor PR-AUC/ROC-AUC | Sin tuning puede sobreajustar; ~10 s | ~10 s |
| KNN | Referencia no paramétrica | Escalado obligatorio | Simple | Costoso; sensible a dimensionalidad | ~0,1 s |

- **Hiperparámetros iniciales** (sin tuning, comparación justa): logística `C=1, max_iter=1000`; árbol `max_depth=6`; RF `n_estimators=400, min_samples_leaf=5, class_weight='balanced_subsample'`; GB por defecto; KNN `n_neighbors=25`.
- **Ranking preliminar por PR-AUC** (holdout): Random Forest 0,453 · Gradient Boosting 0,453 · logística 0,430 · logística balanceada 0,429 · árbol 0,407 · KNN 0,405 (referencia trivial 0,113).
- **Interpretabilidad**: coeficientes de la logística (`poutcome_failure` −1,08 · `emp.var.rate` −0,79 · `poutcome_success` +0,78) e importancias del Random Forest (`euribor3m` 0,21 · `nr.employed` 0,14 · `age` 0,13).
- Los 6 modelos pasan a validación cruzada (§9) por superar claramente a la referencia trivial.

## 9. Estrategia de validación

> Evidencia reproducible en `notebooks/05_validation.ipynb`.

- **División train/test**: 80/20 estratificada, `random_state=42` (32.940 train / 8.236 test). El test **queda reservado** para §10.
- **Estratificación**: sí, por `y`; ~11,27% `yes` en cada fold y en test.
- **Validación cruzada**: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` sobre el train.
- **Prevención de leakage**: preprocesador + modelo como un único `Pipeline`; `cross_validate` re-ajusta todo por fold. Duplicados eliminados antes del split (§6).
- **Métricas evaluadas**: PR-AUC, ROC-AUC, precision, recall, F1.
- **Métrica principal: PR-AUC.** Justificación: con desbalance 7,9:1 el accuracy engaña; PR-AUC resume el desempeño sobre la clase `yes` a lo largo de todos los umbrales y es insensible al gran número de verdaderos negativos.

**Resultados de CV (5 folds, ordenados por PR-AUC):**

| Modelo | PR-AUC (media ± std) | ROC-AUC | Precision | Recall | F1 | t_fit/fold |
| --- | --- | --- | --- | --- | --- | --- |
| Gradient Boosting | **0,447 ± 0,014** | 0,793 | 0,662 | 0,215 | 0,324 | ~15,9 s |
| Random Forest | 0,441 ± 0,016 | 0,786 | 0,410 | 0,589 | 0,484 | ~20,2 s |
| Regresión logística | 0,421 ± 0,013 | 0,760 | 0,699 | 0,182 | 0,289 | ~0,8 s |
| Regresión logística balanceada | 0,420 ± 0,013 | 0,761 | 0,277 | 0,672 | 0,393 | ~0,7 s |
| Árbol de decisión | 0,397 ± 0,017 | 0,775 | 0,644 | 0,214 | 0,321 | ~0,4 s |
| KNN | 0,385 ± 0,014 | 0,753 | 0,623 | 0,208 | 0,312 | ~0,2 s |

- **Estabilidad**: los desvíos de PR-AUC (0,013–0,017) indican desempeño estable entre folds.
- **Umbral**: elegido con probabilidades **out-of-fold** (sin tocar el test). Para Gradient Boosting, el umbral que maximiza F1 es **0,177** (F1 0,493 vs 0,324 con 0,5); un umbral orientado a recall (precision ≥ 0,30) es **0,085** (recall 0,673, precision 0,300).

## 10. Resultados

> Evidencia reproducible en `notebooks/06_results.ipynb` (test reservado, 8.236 filas, una única evaluación).

| Modelo | Umbral | Accuracy | Precision | Recall | F1-score | ROC-AUC | PR-AUC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Dummy (prior) | 0,500 | 0,887 | 0,000 | 0,000 | 0,000 | 0,500 | 0,113 |
| Regresión logística | 0,500 | 0,898 | 0,693 | 0,168 | 0,271 | 0,776 | 0,430 |
| Gradient Boosting | 0,500 | 0,897 | 0,640 | 0,197 | 0,301 | 0,808 | 0,453 |
| Gradient Boosting | 0,177 | 0,873 | 0,451 | 0,605 | 0,517 | 0,808 | 0,453 |

- **Matriz de confusión — GB con umbral 0,177 (F1):** TP 561 · FN 367 · FP 682 · TN 6.626.
  - Recupera el **60,5%** de los clientes que contratarían (561/928) a costa de 682 llamadas inútiles.
  - Bajar el umbral de 0,5 → 0,177 mejora recall **+40,7 pp** y F1 **+21,5 pp**, reduciendo precision −18,9 pp; ROC-AUC y PR-AUC no cambian (el umbral no altera el ordenamiento).
- **Curva Precision-Recall**: el área (PR-AUC ≈ 0,45) supera ampliamente la tasa base (0,113), confirmando capacidad de ranking.
- **Variabilidad CV vs test**: PR-AUC en CV 0,447 (std 0,014) vs 0,453 en test → diferencia −0,006, sin señales de sobreajuste.
- **Interpretación de errores**: los falsos negativos (oportunidades perdidas) son los más costosos; el umbral bajo prioriza reducirlos. El límite superior está dado por las señales disponibles antes de la llamada.

## 11. Modelo seleccionado

> Decisión documentada en `notebooks/06_results.ipynb`, §11.

**Modelo principal: Gradient Boosting** (preprocesamiento de §6, sin tuning formal), con **umbral ajustado** según el objetivo de negocio.

**Modelo interpretable de referencia: Regresión logística** (coeficientes legibles, ~0,8 s/fold).

Criterios considerados (no solo la métrica más alta):

| Criterio | Evidencia |
| --- | --- |
| Métrica principal (PR-AUC) | GB 0,447 en CV (mejor) · 0,453 en test · std 0,014 |
| Estabilidad | Desvíos entre folds muy bajos (≤ 0,017) |
| Falsos positivos/negativos | Umbral configurable: 0,177 (F1) o 0,085 (recall); FP 682 vs FN 367 en el punto F1 |
| Interpretabilidad | Logística > GB/RF; se conserva la logística como referencia |
| Complejidad/costo | GB ~16 s/fold vs logística ~0,8 s/fold |
| Uso previsto | Ranking de clientes por probabilidad → PR-AUC/ROC-AUC relevantes |
| Disponibilidad real | Las 13 features existen antes de llamar (§3) |

**Trade-off central**: no existe un único modelo; se elige **GB** por capacidad de ranking y se documenta el umbral como parámetro de decisión. Random Forest ofrece mayor recall con `balanced` a costa de menor PR-AUC; la logística es preferible si prima la interpretabilidad.

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
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_baseline.ipynb
│   ├── 04_models.ipynb
│   ├── 05_validation.ipynb
│   └── 06_results.ipynb
├── src/
│   ├── preprocessing.py
│   └── models.py
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

Los notebooks estarán numerados según el orden previsto de ejecución. Ejecutarlos en orden y usando un kernel asociado al entorno virtual `.venv`:

```text
notebooks/01_exploratory_analysis.ipynb
notebooks/02_preprocessing.ipynb
notebooks/03_baseline.ipynb
notebooks/04_models.ipynb
notebooks/05_validation.ipynb
notebooks/06_results.ipynb
```

El código reutilizable vive en `src/` y los notebooks lo importan desde la **raíz del repositorio** (`.venv` debe ejecutarse con `jupyter lab` abierto en la raíz). La lógica de preprocesamiento está en `src/preprocessing.py`.

> **Nota:** esta sección deberá actualizarse conforme evolucione la implementación para reflejar el procedimiento real de reproducción.

## 17. Próximos pasos

- Completar la comprensión y documentación del dataset. ✅
- Realizar el análisis exploratorio. ✅
- Establecer el baseline. ✅
- Diseñar el pipeline de preprocesamiento. ✅
- Evaluar modelos bajo una estrategia común de validación. ✅
- Documentar resultados, limitaciones y conclusiones.
- Redactar §12 (limitaciones) y §13 (conclusiones) con la evidencia generada.

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
