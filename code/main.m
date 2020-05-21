clc;
clear;
clear classes;
close all;
pkg load statistics;

## Seleccion de datos para entrenamiento
numClasses = 5;     #cant. de clases
alpha = 0.3; 
minilote = 100;

activacion = {sigmoide(),sigmoide(),sigmoide(),softmax()};

lossMethod = "cross_entropy";
dscMethod = "momentum";

beta = 0.9;
beta2 = 0.99;

layerSize = [5,5,3,5];

## Creacion de la red neuronal
ann = model1(numClasses, alpha, minilote, activacion, layerSize,lossMethod, dscMethod, beta, beta2);

## Creacion de datos
numDatos = 1000;    #cant. de datos de entrenamiento
numDatosVal = 100;  #cant. de datos de validacion
[Xraw,Yraw] = create_data(numDatos,numClasses, 'vertical');
[Xval,Yval] = create_data(numDatosVal,numClasses, 'vertical');

## Se entrena la red y se guardan los pesos finales
ann.train1(Xraw,Yraw,Xval,Yval);

[Xraw,Yraw] = create_data(numDatos,numClasses, 'vertical');
ann.predict1(Xraw,Yraw);
ann.confusion1(Xraw,Yraw);



#{
numDatos = 1000;    #cant. de datos de entrenamiento
numDatosVal = 100;  #cant. de datos de validacion

## Creacion de datos
[Xraw,Yraw] = create_data(numDatos,numClasses, 'vertical');
[Xval,Yval] = create_data(numDatosVal,numClasses, 'vertical');


## Seleccion de valores para entrenamiento
ann.alpha = 0.3;    #tasa de aprendizaje
ann.minilote = 100; #tama�o de minilote
ann.epochs = 3000;  #numero de epocas de entrenamiento
ann.opt = 'momentum';   #opciones de optimizacion: 'adam', 'momentum', 'pure'

## Cantidad de neuronas por capa (4 capas actualmente)
## la cant. de neuronas de la ultima capa = numClasses por default
ann.neurons1 = 5;
ann.neurons2 = 5;
ann.neurons3 = 3;

## Otros valores necesarios para la red (no modificar)
ann.dimensionX = columns(Xraw);
ann.clases = numClasses;


## Se cargan los pesos, si existen
file = "weights.dat";

if (exist(file,"file") == 2)
  ann.load(file);
else
  ann.init();
endif

## Se entrena la red y se guardan los pesos finales
ann.train(Xraw,Yraw,Xval,Yval);
ann.save(file);

clc;
clear;
clear classes;
close all;
pkg load statistics;

## Selecion de parametros para la prediccion
## Usar los mismos utilizados para el entrenamiento
numClasses = 5;
numDatos = 1000;

[Xraw,Yraw] = create_data(numDatos,numClasses, 'vertical');

ann = model();

file = "weights.dat";

if (exist(file,"file") == 2)
  ann.load(file);
  ann.predict(Xraw,Yraw);
  ann.confusion(Xraw,Yraw);
else
  disp("No fue posible encontrar el archivo weights.dat");
endif
#}