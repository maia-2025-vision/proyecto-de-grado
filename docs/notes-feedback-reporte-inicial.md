Feedback de Isaí en el informe inicial.

## Preparación de datos 

En la configuración 2,

 - ¿Cuántos parches de prueba generan? 
 - ¿Se quedan solamente con los parches con animales o todos?
 - ¿Cómo es la distribución de los datos en comparativa de train vs test?


## Descripción de proceso de construcción de modelos


Score: 30 out of 35 points

Hace falta mencionar hiperparámetros usados tales como learning rate, scheduler, número de epocas, etc. 
- ¿En FasterRCNN que valores de anchors usan? (teo: no sé a qué se refiere esto...)
- ¿Usan presos preentrenados?


# Resultados obtenidos


Tienen variaciones en configuraciones (fracción mínima visible), 
diferentes backbones (dla, resnet 50, resnet 101) pero solo presentan 2 resultados finales. 

  - ¿Qué pasó con las diferentes configuraciones que mencionan?
  - Hace falta mostrar resultados cualitativos del modelo. 


# Análisis de resultados obtenidos:

Score: *14 out of 20 points* 

  * El análisis sería mucho más rico si se incluyen resultados cualitativos y su respectiva discusión.
  * Hace falta comparación con la línea base (HerdNet).
  * ¿Cómo obtienen las métricas de FasterRCNN?
  * ¿Convierten las bboxes en centroides y usan el mismo script de test que herdnet o cómo lo están haciendo?


# Presentación

Usar captions tanto para figuras como para tablas
