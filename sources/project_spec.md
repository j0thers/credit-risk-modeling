 PROYECTO DE FINANZAS CUANTITATIVAS
Modelización estadística y evaluación probabilística del riesgo de incumplimiento crediticio
Área: Estadística aplicada – Finanzas cuantitativas – Machine Learning
Enfoque: Modelización del riesgo de crédito (Credit Risk Modeling)


1. PLANTEAMIENTO DEL PROYECTO
1.1. Descripción general
El proyecto consiste en desarrollar un sistema de modelización estadística capaz de estimar la probabilidad de que un cliente incumpla sus obligaciones crediticias.
Se utilizarán variables demográficas, financieras y de comportamiento de pago para construir y comparar diferentes modelos estadísticos y algoritmos de aprendizaje automático.
El propósito no será únicamente identificar qué modelo clasifica mejor a los clientes, sino también determinar qué tan confiables son las probabilidades estimadas y cómo podrían utilizarse para cuantificar el riesgo financiero.
1.2. Pregunta de investigación
¿En qué medida las características demográficas, financieras y el comportamiento histórico de pago permiten estimar la probabilidad de incumplimiento crediticio, y qué modelos estadísticos ofrecen las predicciones más precisas y mejor calibradas?
1.3. Objetivo general
Desarrollar y evaluar modelos estadísticos y algoritmos de aprendizaje automático para estimar la probabilidad de incumplimiento crediticio, comparando su capacidad predictiva, interpretabilidad y calibración probabilística.
1.4. Objetivos específicos
1. Identificar las principales características financieras y de comportamiento asociadas con el incumplimiento crediticio.
2. Construir un modelo de regresión logística que permita interpretar la relación entre las variables explicativas y el incumplimiento.
3. Desarrollar modelos de aprendizaje automático para estimar la probabilidad de incumplimiento.
4. Comparar el desempeño de los modelos mediante métricas estadísticas de clasificación.
5. Evaluar la calibración de las probabilidades estimadas.
6. Explorar la utilidad de las probabilidades obtenidas para calcular pérdidas crediticias esperadas bajo distintos escenarios.

2. DATASETS Y FUENTES DE INFORMACIÓN
2.1. Dataset principal: Default of Credit Card Clients
Este será el único dataset obligatorio para desarrollar el proyecto principal.
- Institución: UCI Machine Learning Repository.
- Nombre: Default of Credit Card Clients.
- Observaciones: 30.000 clientes.
- Variables explicativas: 23.
- Variable objetivo: incumplimiento del pago en el siguiente mes.
- Contexto: clientes de tarjetas de crédito de Taiwán.
- Periodo de referencia: abril a septiembre de 2005.
Enlace de descarga:
https://archive.ics.uci.edu/dataset/350/default%2Bof%2Bcredit+card+clients
DOI: https://doi.org/10.24432/C55S3H
2.2. Variables disponibles
A. Variables demográficas y personales
- SEX: sexo del cliente.
- EDUCATION: nivel educativo.
- MARRIAGE: estado civil.
- AGE: edad.
B. Variables financieras
- LIMIT_BAL: límite de crédito.
- BILL_AMT1 a BILL_AMT6: montos facturados durante los seis meses observados.
- PAY_AMT1 a PAY_AMT6: montos pagados durante esos meses.
C. Historial de comportamiento de pago
- PAY_0: estado de pago de septiembre.
- PAY_2: estado de pago de agosto.
- PAY_3: estado de pago de julio.
- PAY_4: estado de pago de junio.
- PAY_5: estado de pago de mayo.
- PAY_6: estado de pago de abril.
D. Variable objetivo
- default.payment.next.month: indica si el cliente incumplió sus obligaciones de pago en el siguiente mes.
Esta será la variable dependiente del proyecto.
2.3. Datasets y fuentes complementarias
No es necesario descargar múltiples bases de datos para construir el modelo principal. Sin embargo, estas fuentes pueden complementar la investigación:
FuenteUtilidadPrioridadUCI Default of Credit Card ClientsConstrucción y evaluación de modelos de riesgo crediticioObligatoriaDocumentación de UCIComprender las variables y el origen de los datosObligatoriaArtículo de Yeh y Lien (2009)Referencia metodológica y contexto del datasetRecomendadaFREDConsultar indicadores macroeconómicos y explorar su relación con el riesgo crediticioOpcionalBanco de la RepúblicaContextualizar el riesgo crediticio desde una perspectiva colombianaOpcionalSuperintendencia Financiera de ColombiaConsultar indicadores y estadísticas del sistema financiero colombianoOpcional
Referencias complementarias:
- Artículo original: https://doi.org/10.1016/j.eswa.2007.12.020
- FRED: https://fred.stlouisfed.org/
- Banco de la República: https://www.banrep.gov.co/
- Superintendencia Financiera: https://www.superfinanciera.gov.co/
Nota metodológica: el dataset principal corresponde a clientes de Taiwán en 2005. Por tanto, los resultados no deben interpretarse como estimaciones directamente aplicables al mercado crediticio colombiano. Las fuentes colombianas servirán para contextualizar, no para mezclar indiscriminadamente poblaciones y periodos diferentes.

3. HOJA DE RUTA DEL PROYECTO
FASE 1. Adquisición y comprensión de los datos
Semana 1
Actividades:
1. Descargar el dataset de UCI.
2. Identificar la estructura de la base de datos.
3. Examinar los tipos de variables.
4. Identificar valores faltantes y registros duplicados.
5. Analizar las categorías de las variables.
6. Documentar el origen y las limitaciones de los datos.
Herramientas:
- Python.
- Pandas.
- NumPy.
- Jupyter Notebook.
Entregable: documentación inicial y auditoría de calidad de los datos.
FASE 2. Análisis exploratorio de datos (EDA)
Semana 2
Actividades:
1. Calcular la proporción de clientes que incumplieron sus obligaciones.
2. Analizar la distribución de edades y límites de crédito.
3. Examinar los historiales de pago.
4. Comparar el comportamiento financiero entre clientes cumplidos e incumplidos.
5. Analizar correlaciones entre variables.
6. Aplicar pruebas de asociación cuando corresponda.
Técnicas:
- Estadística descriptiva.
- Tablas de contingencia.
- Pruebas chi-cuadrado.
- Correlaciones.
- Histogramas y diagramas de caja.
- Visualización de distribuciones.
Entregable: informe exploratorio con tablas, visualizaciones y hallazgos preliminares.
FASE 3. Modelización estadística: regresión logística
Semana 3
Actividades:
1. Definir la variable objetivo.
2. Seleccionar las variables explicativas.
3. Preparar los datos para el modelamiento.
4. Ajustar una regresión logística.
5. Interpretar los coeficientes.
6. Calcular odds ratios.
7. Evaluar significancia estadística.
8. Examinar posibles problemas de multicolinealidad.
Modelo:
- Regresión logística binaria.
Entregable: primer modelo estadístico interpretable y análisis de sus resultados.
FASE 4. Diagnóstico y validación estadística
Semana 4
Actividades:
1. Evaluar el desempeño inicial del modelo.
2. Analizar residuos y diagnósticos pertinentes.
3. Examinar la influencia de variables individuales.
4. Evaluar posibles interacciones.
5. Comparar especificaciones alternativas.
6. Realizar validación cruzada estratificada.
Entregable: modelo logístico revisado y documentación de sus limitaciones.
FASE 5. Machine Learning aplicado al riesgo crediticio
Semana 5 y 6
Desarrollar modelos alternativos:
1. Árbol de decisión.
2. Random Forest.
3. XGBoost.
Actividades:
- Preprocesamiento de variables.
- Ajuste de hiperparámetros.
- Selección de características.
- Comparación entre algoritmos.
- Prevención de sobreajuste.
Entregable: modelos predictivos entrenados y comparables.
FASE 6. Evaluación comparativa
Semana 7
Comparar los modelos mediante:
MétricaPropósitoAccuracyProporción total de clasificaciones correctasPrecisionProporción de predicciones positivas correctasRecallCapacidad para identificar incumplimientosF1-scoreEquilibrio entre precision y recallROC-AUCCapacidad de discriminaciónPR-AUCDesempeño en la identificación de la clase positiva
También se realizará un experimento de ablación:
- Modelo A: variables demográficas.
- Modelo B: historial de pagos.
- Modelo C: variables financieras.
- Modelo D: todas las variables.
El objetivo será determinar qué grupos de variables aportan mayor información predictiva.
Entregable: tabla comparativa y análisis de los resultados.
FASE 7. Calibración y estimación de pérdidas esperadas
Semana 8
Esta fase permitirá profundizar en la dimensión financiera del proyecto.
Actividades:
1. Obtener probabilidades individuales de incumplimiento.
2. Evaluar la calibración de las probabilidades.
3. Construir curvas de calibración.
4. Calcular Brier Score.
5. Calcular Log Loss.
6. Explorar técnicas de calibración como Platt Scaling e Isotonic Regression.
7. Simular escenarios de pérdidas crediticias esperadas.
Conceptos:
- Probability of Default (PD).
- Loss Given Default (LGD).
- Exposure at Default (EAD).
- Expected Loss (EL).
Fórmula:
EL = PD × LGD × EAD
Ejemplo hipotético:
- PD = 15 %.
- LGD = 60 %.
- EAD = $2.000.000.
Pérdida esperada:
EL = 0,15 × 0,60 × 2.000.000 = $180.000.
Importante: el dataset permite estimar la probabilidad de incumplimiento para el horizonte del siguiente mes, pero no proporciona directamente los datos necesarios para estimar LGD y EAD empíricamente. Estos dos componentes se trabajarán mediante escenarios hipotéticos claramente identificados.
FASE 8. Consolidación y presentación
Semana 9 y 10
Actividades:
1. Seleccionar los modelos finales.
2. Organizar los resultados.
3. Elaborar conclusiones estadísticas y financieras.
4. Redactar el informe técnico.
5. Documentar el código.
6. Crear el repositorio de GitHub.
7. Preparar una presentación del proyecto.
Entregable final: proyecto documentado, reproducible y publicado en GitHub.

4. METODOLOGÍA DE VALIDACIÓN
Para evitar conclusiones artificialmente optimistas, se utilizará una metodología de evaluación rigurosa.
División de datos recomendada:
- Entrenamiento: 70 %.
- Validación: 15 %.
- Prueba: 15 %.
Se empleará estratificación para conservar aproximadamente la proporción de clientes incumplidos y no incumplidos.
La validación cruzada estratificada se aplicará dentro del conjunto de entrenamiento.
Advertencia: aunque el dataset contiene seis meses de historial financiero, cada cliente aparece una sola vez. Por ello, no debe presentarse una validación temporal walk-forward como si existieran múltiples observaciones mensuales independientes por cliente.
También se deberá:
- Excluir el identificador del cliente como predictor.
- Evitar fuga de información (data leakage).
- Ajustar imputaciones, escalamiento y selección de variables exclusivamente con los datos de entrenamiento.
- Evaluar el conjunto de prueba una sola vez al finalizar el proceso de selección.

5. HERRAMIENTAS Y TECNOLOGÍAS
HerramientaAplicaciónPythonLenguaje principalPandasManipulación de datosNumPyOperaciones numéricasMatplotlibVisualizaciónSeabornAnálisis gráficoSciPyPruebas estadísticasStatsmodelsModelización estadísticaScikit-learnMachine Learning y evaluaciónXGBoostModelos de boostingJupyter NotebookExperimentaciónGit y GitHubControl de versiones y portafolio

6. ESTRUCTURA DEL REPOSITORIO
Nombre sugerido: credit-risk-modeling
credit-risk-modeling/
│
├── data/
│   └── README.md
│
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_logistic_regression.ipynb
│   ├── 04_machine_learning_models.ipynb
│   └── 05_model_evaluation.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── models.py
│   └── evaluation.py
│
├── figures/
│
├── reports/
│   └── technical_report.pdf
│
├── tests/
│
├── requirements.txt
└── README.md

7. RESULTADOS ESPERADOS
Al finalizar el proyecto se espera disponer de:
1. Un análisis exploratorio completo del comportamiento crediticio.
2. Un modelo de regresión logística interpretable.
3. Modelos de Machine Learning entrenados y evaluados.
4. Una comparación rigurosa de desempeño predictivo.
5. Un análisis de calibración de probabilidades.
6. Una estimación de pérdidas esperadas mediante escenarios.
7. Un informe técnico con conclusiones estadísticas.
8. Un repositorio reproducible en GitHub.
Adicionalmente, se recomienda realizar una auditoría de desempeño por grupos demográficos. Esto permitirá identificar posibles diferencias en errores y calibración, evitando interpretar una buena métrica global como garantía de equidad.

8. CRITERIOS DE CALIDAD DEL PROYECTO
El proyecto deberá cumplir los siguientes criterios:
- Justificación estadística de las decisiones metodológicas.
- Interpretación de los resultados, no solamente presentación de métricas.
- Comparación con modelos de referencia.
- Validación adecuada y prevención de fuga de información.
- Reproducibilidad del código.
- Reconocimiento explícito de las limitaciones del dataset.
- Diferenciación entre asociación estadística y causalidad.
- Documentación clara de los procedimientos.

9. REFERENCIAS BIBLIOGRÁFICAS
1. Yeh, I. C., & Lien, C. H. (2009). The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients. Expert Systems with Applications.
2. James, G., Witten, D., Hastie, T., & Tibshirani, R. An Introduction to Statistical Learning.
3. Hastie, T., Tibshirani, R., & Friedman, J. The Elements of Statistical Learning.
4. Tsay, R. S. Analysis of Financial Time Series.
5. Hull, J. C. Risk Management and Financial Institutions.

10. ALCANCE FINAL
Este proyecto constituye una primera aproximación a la modelización cuantitativa del riesgo financiero.
Su importancia radica en integrar tres dimensiones fundamentales:
Estadística: inferencia, regresión, probabilidad y evaluación de modelos.
Machine Learning: algoritmos predictivos, selección de variables, optimización y calibración.
Finanzas cuantitativas: probabilidad de incumplimiento, riesgo crediticio y pérdidas esperadas.
El resultado no será simplemente un ejercicio de clasificación, sino un estudio aplicado que demuestre la capacidad de construir, interpretar y evaluar modelos probabilísticos para resolver un problema financiero real.
Título definitivo sugerido:
Modelización estadística y evaluación probabilística del riesgo de incumplimiento crediticio: comparación entre modelos paramétricos y algoritmos de aprendizaje automático.