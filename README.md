# p1 IRP
ALFARO GONZALEZ OSVALDO JESUS
FLORES HERRERA ALEJANDRO
LOPEZ MEJIA JEAN PAUL
SANCHEZ MADRIZ SEBASTIAN

Instrucciones de ejecución:

1. Se debe ejecutar el archivo train.m, se deben modificar ciertos parametros dependiendo lo que se quiera predecir,
en la parte de Selección de datos para entrenamiento numClasses indica la cantidad de clases deseadas a crear, numDatos 
la cantidad total de datos a entrenar y numDatosVal la cantidad de datos de validación. En este mismo código está la parte de
creación de datos, en esta parte se eligen entre ¨horizontal, vertical, pie, curved y spirals¨. Para finalizar se indican los valores para el entrenamiento 
y se indica el tipo de optimización que desee, por último se ejecuta este código.
2. Si se desea hacer cambios en las capas de activación se deben cambiar en el model.m de lo contrario está estructurado en 4 capas, 3 sigmoides y
softmax cross entropy para la última.
3. De tercera instancia se procede a ejecutar el archivo predict.m tomando encuenta que también en este código se debe cambiar el tipo de
paramentros de la predicción usada en el paso anterior, es decir si se quizo entrenar ¨horizontal¨ en este código se debe indicar que la predicción tambión es horizontal,
una vez se ejecute este código aparecerán las gráficas con las clases y sus predicciones.
