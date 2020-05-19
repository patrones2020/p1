 
clc;
clear;
clear classes;
close all;
pkg load statistics;

numClasses = 5;
numDatos = 1000;
[Xraw,Yraw] = create_data(numDatos,numClasses, 'vertical');

ann = model();

ann.clases = numClasses;
ann.alpha = 0.3;
ann.minilote = 100;
ann.epochs = 1500;
ann.dimensionX = columns(Xraw);

## Cantidad de neuronas por capa (4 capas actualmente)
ann.neurons1 = 5;
ann.neurons2 = 5;
ann.neurons3 = 3;
#la cant. de neuronas de la ultima capa = numClasses por default


file = "weights.dat";

if (exist(file,"file") == 2)
  ann.load(file);
else
  ann.init();
endif

ann.train(Xraw,Yraw);

ann.save(file);