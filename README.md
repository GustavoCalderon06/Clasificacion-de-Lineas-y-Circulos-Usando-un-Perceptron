# Proyecto de Clasificación con Perceptrón

Este proyecto implementa un **perceptrón simple** para clasificar patrones en imágenes bidimensionales. Utilizando Python y bibliotecas de aprendizaje automático, entrenamos el modelo para identificar líneas y círculos en un conjunto de datos de imágenes generadas.

## Descripción

El objetivo es demostrar la eficacia de un perceptrón simple en tareas de clasificación básicas y explorar su rendimiento. Mediante múltiples ejecuciones y análisis de la matriz de confusión, evaluamos la precisión y la capacidad de aprendizaje del modelo.

### Características

- Clasificación binaria de patrones: **líneas** vs. **círculos**
- Entrenamiento supervisado con un enfoque de clasificación binaria
- Visualización de resultados y análisis de rendimiento

### Funciones de Activación

Para asegurar el correcto funcionamiento del perceptrón, se implementaron dos tipos de funciones de activación:
- **Función Escalón**: Utilizada en la capa oculta, genera una salida de 1 o 0 dependiendo de si la suma ponderada de las entradas supera un umbral.
- **Función Softmax**: Aplicada en la capa de salida, convierte las salidas en probabilidades, facilitando la evaluación de pertenencia a cada categoría y permitiendo la clasificación multiclase.

## Tecnologías Utilizadas

- **Python**: Lenguaje principal de implementación
- **NumPy y Matplotlib**: Para manejo de datos y visualización
- **Scikit-Learn**: Para evaluación de métricas y preprocesamiento
