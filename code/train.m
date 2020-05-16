clc;
clear;
clear classes;
close all;
pkg load statistics;

numClasses = 5;
numDatos = 1000;
[Xraw,Yraw] = create_data(numDatos,numClasses, 'radial');

ann = model();

ann.clases = numClasses;
ann.alpha = 0.3;
ann.minilote = 100;
ann.epochs = 2500;
ann.neurons1 = 5;
ann.neurons2 = 3;
ann.dimensionX = columns(Xraw);

file = "weights.dat";

if (exist(file,"file") == 2)
  ann.load(file);
else
  ann.init();
endif

ann.train(Xraw,Yraw);

ann.save(file);

