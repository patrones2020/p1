clc;
clear;
clear classes;
close all;
pkg load statistics;

## Selecion de parametros para la prediccion
## Usar los mismos utilizados para el entrenamiento
numClasses = 5;
numDatos = 1000;

[Xraw,Yraw] = create_data(numDatos,numClasses, 'spirals');

ann = model();

file = "weights.dat";

if (exist(file,"file") == 2)
  ann.load(file);
  ann.predict(Xraw,Yraw);
  ann.confusion(Xraw,Yraw);
else
  disp("No fue posible encontrar el archivo weights.dat");
endif
