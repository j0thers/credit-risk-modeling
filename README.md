# Modelización del riesgo de incumplimiento crediticio

## Planteamiento y Resultados Principales

El proyecto estima la probabilidad de incumplimiento al mes siguiente en clientes de tarjetas de crédito. Usa 30.000 registros del conjunto UCI de Taiwán con historial de abril a septiembre de 2005. La logística M5 sirve para inferencia e interpretación. XGBoost isotónico es la PD elegida entre calibradores con validación. Los datos y la referencia se describen en el [registro de UCI](https://doi.org/10.24432/C55S3H).

### Evaluación en prueba

Prueba estratificada de 4.500 clientes, 996 con incumplimiento. Se conservan los decimales de las salidas archivadas. PR-AUC se reporta como precisión promedio (AP). ECE figura con bins uniformes y por cuantiles.

| Modelo | Método | ROC-AUC | AP | Log Loss | Brier | ECE uniforme | ECE cuantiles |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Logística M5 | cruda | 0.7294188168197906 | 0.5021438417571005 | 0.4605764198065483 | 0.14320582234349113 | 0.03219213335604169 | 0.03133260216802696 |
| Árbol | cruda | 0.7628314341842255 | 0.5262029992799963 | 0.44460428969782767 | 0.1385178476537195 | 0.021675388786908265 | 0.02042212495164716 |
| Random Forest | cruda | 0.7817061625497423 | 0.5600904984503189 | 0.428339443189576 | 0.13443870374731062 | 0.010662440780186154 | 0.0074637569125647635 |
| XGBoost | cruda | 0.7834374025783499 | 0.5640310442030501 | 0.42713753861056236 | 0.13412214113816873 | 0.010076360120127605 | 0.013886648738135896 |
| XGBoost | sigmoid | 0.7834374025783499 | 0.5640310442030501 | 0.43126527687227634 | 0.13513665782988707 | 0.02057416515199392 | 0.030301000194354426 |
| XGBoost | isotonic | 0.7803670446626689 | 0.5352022425711601 | 0.442940405764039 | 0.13460809897138679 | 0.009719856012581846 | 0.011448891550693804 |
| Random Forest | sigmoid | 0.7817061625497423 | 0.5600904984503189 | 0.4317240021137014 | 0.13525332212884547 | 0.020092264271667045 | 0.02351334149530483 |
| Random Forest | isotonic | 0.7804113142065982 | 0.536968135252119 | 0.43684075464814737 | 0.13480965270464587 | 0.00992316842430148 | 0.01042374999527227 |
| Logística M5 | sigmoid | 0.7294188168197906 | 0.5021438417571005 | 0.45398969691482716 | 0.14227750423550195 | 0.017313445756245965 | 0.019658589716460013 |
| Logística M5 | isotonic | 0.7288989003961049 | 0.4842806516374539 | 0.46687865819101515 | 0.14161248287478484 | 0.0112896060519281 | 0.007641668174284574 |


XGBoost crudo lideró en validación por ROC-AUC y en prueba obtuvo los menores Brier y Log Loss. La elección de isotónica se hizo en validación entre las opciones calibradas, antes de evaluarlas en prueba. En el contraste pareado XGBoost–bosque, el intervalo del 95 % para la diferencia de ROC-AUC incluye cero. El historial de pagos es el bloque más informativo en ablación. La auditoría demográfica muestra diferencias descriptivas en PD media y tasas de error. Los resultados describen asociaciones en Taiwán en 2005. Transferir el modelo a Colombia exige validación externa con datos locales y recalibración o reestimación. La pérdida esperada es una simulación con LGD y exposición hipotéticas.

## Estructura auditada

El árbol enumera los archivos del workspace y omite la caché generada por Python. La carpeta data y el notebook exploratorio residen dentro de los ZIP históricos. En la raíz no hay checkout de Git, carpeta tests ni segundo notebook de preprocesamiento.

```text
credit-risk-modeling/
├── AGENTS.md
├── figures/
│   ├── 05_logistic_diagnostics_residuals.png
│   ├── 06_logistic_influence_cooks.png
│   ├── 07_ml_feature_importance.png
│   ├── 08_test_roc_pr_curves.png
│   ├── 09_test_confusion_matrices.png
│   ├── 10_ablation_study_comparison.png
│   ├── 11_calibration_curves_comparison.png
│   ├── 12_demographic_fairness_audit.png
│   └── 13_expected_loss_scenarios.png
├── notebooks/
│   ├── 03_logistic_regression.ipynb
│   ├── 04_machine_learning_models.ipynb
│   ├── 05_model_evaluation.ipynb
│   ├── phase4_logistic.py
│   └── phase7_calibration.py
├── README.md
├── reports/
│   ├── 06_ablation_deltas.csv
│   ├── 06_ablation_metrics.csv
│   ├── 06_classification_metrics.csv
│   ├── 06_confusion_normalized.csv
│   ├── 06_frozen_protocol.json
│   ├── 06_paired_auc.csv
│   ├── 06_probabilities_test.csv
│   ├── 06_probabilities_val.csv
│   ├── 06_probability_metrics.csv
│   ├── 07_calibration_metrics.csv
│   ├── 07_demographic_audit.csv
│   ├── 07_expected_loss_individual.csv
│   ├── 07_expected_loss_scenarios.csv
│   ├── 07_protocol.json
│   ├── 07_validation_ranking.csv
│   ├── 07_validation_selection.csv
│   ├── executive_storyboard.md
│   ├── phase4_results.json
│   ├── phase5_models/
│   │   ├── forest.joblib
│   │   ├── hist_gradient_boosting.joblib
│   │   ├── logistic.joblib
│   │   └── tree.joblib
│   ├── phase5_results.json
│   ├── phase6_models/
│   │   ├── xgboost.joblib
│   │   └── xgboost_cv.json
│   └── technical_report.tex
├── requirements.txt
├── sources/
│   ├── default of credit card clients.xls
│   ├── phase1_deliverables.zip
│   ├── phase2_deliverables.zip
│   ├── project_spec.md
│   └── ssrn-5398933.pdf
└── src/
    ├── evaluation.py
    ├── models.py
    └── preprocessing.py
```

### Figuras disponibles

- [05_logistic_diagnostics_residuals.png](figures/05_logistic_diagnostics_residuals.png)
- [06_logistic_influence_cooks.png](figures/06_logistic_influence_cooks.png)
- [07_ml_feature_importance.png](figures/07_ml_feature_importance.png)
- [08_test_roc_pr_curves.png](figures/08_test_roc_pr_curves.png)
- [09_test_confusion_matrices.png](figures/09_test_confusion_matrices.png)
- [10_ablation_study_comparison.png](figures/10_ablation_study_comparison.png)
- [11_calibration_curves_comparison.png](figures/11_calibration_curves_comparison.png)
- [12_demographic_fairness_audit.png](figures/12_demographic_fairness_audit.png)
- [13_expected_loss_scenarios.png](figures/13_expected_loss_scenarios.png)

El ZIP de análisis exploratorio conserva por separado las figuras 01 a 04 y sus tablas. Sus archivos internos se pueden inspeccionar con `python -m zipfile -l sources/phase2_deliverables.zip`.

## Reproducción

1. Cuando el repositorio esté publicado, sustituye `<URL_GITHUB>` por la URL del repositorio.

   ```powershell
   git clone <URL_GITHUB> credit-risk-modeling
   cd credit-risk-modeling
   py -3.12 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install -r requirements.txt
   python -m ipykernel install --sys-prefix --name credit-risk-modeling
   ```

2. Conserva los ZIP en `sources/`. La ruta principal lee la tabla limpia y los índices archivados directamente del ZIP de auditoría. Ambos archivos comprimidos incluyen scripts y salidas. El archivo de análisis exploratorio documenta un entorno histórico propio y el uso de etiquetas de toda la muestra en forma agregada, lo cual limita la independencia de prueba.
3. Desde la raíz, abre los notebooks en VS Code con el kernel `credit-risk-modeling`. Ejecuta `03_logistic_regression.ipynb` para la base inferencial y `python -m notebooks.phase4_logistic` para los diagnósticos y el modelo final. Los módulos de `src/` se importan desde los puntos de entrada y se encargan respectivamente de preparación, ajuste y métricas.
4. Para reproducir la evaluación comparativa y la calibración, ejecuta `05_model_evaluation.ipynb` en orden con [phase5_results.json](reports/phase5_results.json) y los modelos de `reports/phase5_models/`. El último bloque llama a `run_phase7`. Para repetir únicamente la calibración, ejecuta `python -m notebooks.phase7_calibration` con las probabilidades congeladas de la evaluación comparativa.
5. Para volver a entrenar los modelos predictivos, ejecuta `04_machine_learning_models.ipynb` entre los pasos 3 y 4 en una copia de trabajo. El notebook actual detecta XGBoost si está instalado. El ajuste archivado para el análisis de importancia empleó HistGradientBoosting. Un nuevo ajuste puede sobrescribir artefactos históricos y cambiar el origen del XGBoost evaluado en prueba. Para replicar las cifras archivadas, usa los modelos y salidas conservados.
6. `python -m pytest` ejecuta la suite cuando se incorpora al checkout. El workspace auditado carece de pruebas versionadas; en este estado pytest debería reportar cero pruebas recolectadas.

## Informe y presentación

- [Informe técnico LaTeX](reports/technical_report.tex). Compila desde la raíz con `pdflatex -interaction=nonstopmode -halt-on-error reports/technical_report.tex`.
- [Guion ejecutivo de diez láminas](reports/executive_storyboard.md).
- [Protocolo congelado y huellas de entradas](reports/07_protocol.json).
