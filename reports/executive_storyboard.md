# Guion de presentación de resultados empíricos

## 1. Pregunta de investigación y población

- PD de incumplimiento a un mes en tarjetas de crédito.
- 30.000 clientes en Taiwán con historia de abril a septiembre de 2005.
- Objetivos diferenciados de interpretación y estimación probabilística.

Figura a proyectar: [05_logistic_diagnostics_residuals.png](../figures/05_logistic_diagnostics_residuals.png)

## 2. Diseño de validación

- Train / val / test de 21.000 / 4.500 / 4.500, semilla 42.
- Preprocesamiento y ajuste en entrenamiento, con ID excluido.
- EDA global histórica limita la independencia estricta de prueba.

Figura a proyectar: [08_test_roc_pr_curves.png](../figures/08_test_roc_pr_curves.png)

## 3. Logística M5 para interpretación

- La especificación incorpora dos interacciones y curvatura del límite.
- Los LRT de omisión respaldan los tres términos adicionales.
- Los OR son condicionales y los IC Wald son nominales tras selección.

Figura a proyectar: [05_logistic_diagnostics_residuals.png](../figures/05_logistic_diagnostics_residuals.png)

## 4. Colinealidad e influencia

- BILL_AMT2 alcanza VIF 27.602493583173345 con los seis saldos.
- El VIF final de BILL_AMT1 es 2.439754131609642.
- La distancia de Cook permite localizar las observaciones más influyentes.

Figura a proyectar: [06_logistic_influence_cooks.png](../figures/06_logistic_influence_cooks.png)

## 5. Algoritmos y variables predictivas

- El ajuste predictivo archivado incluyó árbol, bosque e HistGradientBoosting junto con la logística.
- La comparación en prueba incluyó XGBoost con selección interna de hiperparámetros.
- La figura presenta la importancia de predictores en validación.

Figura a proyectar: [07_ml_feature_importance.png](../figures/07_ml_feature_importance.png)

## 6. Discriminación y umbrales en prueba

- XGBoost crudo alcanza ROC-AUC 0.7834374025783499.
- El IC pareado de la diferencia con el bosque incluye cero.
- El corte estándar y el Youden congelado representan distintos intercambios de error.

Figura a proyectar: [09_test_confusion_matrices.png](../figures/09_test_confusion_matrices.png)

## 7. Ablación e interpretación financiera

- El estado de pago es el bloque aislado más informativo.
- Retirarlo de XGBoost reduce AP en 0.11632586171961767.
- La persistencia de mora es señal predictiva observacional.

Figura a proyectar: [10_ablation_study_comparison.png](../figures/10_ablation_study_comparison.png)

## 8. PD seleccionada y calibración

- XGBoost isotónico gana por Brier entre calibradores en validación.
- En prueba reduce los dos ECE de XGBoost.
- La PD cruda conserva mejor Brier, Log Loss, ROC-AUC y AP.

Figura a proyectar: [11_calibration_curves_comparison.png](../figures/11_calibration_curves_comparison.png)

## 9. Auditoría por subgrupos

- Los hombres presentan subestimación media y mayor FPR y FNR que las mujeres.
- Mayores de 45 años muestran la FPR más alta entre grupos etarios.
- Educación y estado civil incluyen grupos pequeños con estimaciones inestables.

Figura a proyectar: [12_demographic_fairness_audit.png](../figures/12_demographic_fairness_audit.png)

## 10. Pérdida esperada y transferencia

- La PD empírica estima el incumplimiento a un mes.
- Los escenarios Base, Adverso y Severo usan LGD y exposición hipotéticas.
- La aplicación en Colombia exige datos locales, validación externa y recalibración o reestimación.

Figura a proyectar: [13_expected_loss_scenarios.png](../figures/13_expected_loss_scenarios.png)
