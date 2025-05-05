El proyecto es ejecutado en windows utilizando un ambiente virtual creado con pyenv con python 3.12.6
Las librerías necesarias son:
- mediapipe
- pandas
- opencv-python

Primero se realiza la creación del dataset generado 10 muestras de 16 frames cada una por cada categoría de gesto a detectar.

Luego se extraen las coordenadas de cada imagen para armar el dataset que sirve de input para la red neuronal