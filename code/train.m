clc;
clear;
clear classes;
close all;
pkg load statistics;

## Seleccion de datos para entrenamiento
numClasses = 5;     #cant. de clases
numDatos = 1000;    #cant. de datos de entrenamiento
numDatosVal = 100;  #cant. de datos de validacion

## Creacion de datos
[Xraw,Yraw] = create_data(numDatos,numClasses, 'pie');
[Xval,Yval] = create_data(numDatosVal,numClasses, 'pie');

## Creacion de la red neuronal
ann = model();

## Seleccion de valores para entrenamiento
ann.alpha = 0.3;    #tasa de aprendizaje
ann.minilote = 100; #tamaño de minilote
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

