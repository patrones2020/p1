# p1 IRP
ALFARO GONZALEZ OSVALDO JESUS
FLORES HERRERA ALEJANDRO
LOPEZ MEJIA JEAN PAUL
SANCHEZ MADRIZ SEBASTIAN

Instrucciones de ejecución:

1. Se debe ejecutar el archivo train.m, se deben modificarciertos parametros dependiendo lo que se quiera predecir,
en la parte de Seleccion de datos para entrenamiento numClasses indica la cantidad de clases deseadas a crear, numDatos 
la cantidad total de datos a entrenar y numDatosVal la cantidad de datos de validacion. En este mismo codigo esta la parte de
creacion de datos, en esta parte se eligen entre ¨horizontal, vertical, pie, curved y spiral¨. Para finalizar se indican los valores para el entrenamiento 
y se indica el tipo de optimizacion que desee, por ultimo se ejecuta este codigo.
2. Si se desea hacer cambios en las capas de activacion se deben cambiar en el model.m de lo contrario está estructurado en 4 capas, 3 sigmoides y
softmac cross entrophy para la última.
3. De segunda instancia se procede a ejecutar el archivo predict.m tomando encuenta que tambien en este codigo se debe cambiar el tipo de
paramentros de la prediccion usada en el paso anterior, es decir si se quizo enrenar ¨horizontal¨ en este codigo se debe indicar que la prediccion tambien es horizontal,
una vez se ejecute este codigo apareceran las graficas con las clases y sus predicciones.
