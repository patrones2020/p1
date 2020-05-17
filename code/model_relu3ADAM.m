##Prueba de red neuronal de 3 capas con ADAM
#Prueba de red neuronal con 3 caps
clc;
clear;
clear -classes;
pkg load statistics

## Valores para el entrenamiento
alpha = 0.04;   #tasa de aprendizaje
epocas = 1000;  #numero de epocas
m = 100;        #tama�o de mini lote
clases = 3;     #cantidad de clases
numDatos = 1000;#cantidad de datos de set de entrenamiento

## Creacion de datos, carga las matrices usadas sin momentum
load("XYtrains.dat")

## Capa 1
l1a = fullyConnectedBiased(); #combinacion
l1b = relu();             #activacion
neurons1 = 5;                 #cantidad de neuronas

## Capa 2
l2a = fullyConnectedBiased(); #combinacion
l2b = relu();             #activacion
neurons2 = 3;            


## Capa 3
l3a = fullyConnectedBiased(); #combinacion
l3b = sigmoide(); #activacion
neurons3 = clases;   #la ultima capa debe tener #neuronas = #clases


## Capa de calculo de error
l4 = cross_entropy();

## ADAM
beta = 0.9;
beta2 = 0.99;

mtm1 = ADAM(beta, beta2);
mtm2 = ADAM(beta, beta2);
mtm3 = ADAM(beta, beta2);
beta = 0.9;

## Se inicializan las matrices de pesos
dimension = columns(Xraw);

W1 = rand(neurons1, dimension + 1); #se suma 1 por el sesgo
W2 = rand(neurons2, neurons1 + 1); #las columnas siempre es: cant. neuronas
                                   #anteriores + 1
W3 = rand(neurons3, neurons2 + 1);
                                   
## Para el plot del error
Jacumulados = [];
numEpocas = [];

##Vectores del momentum
v01 = rand(5,3);
v02 = rand(3,6);
v03 = rand(3,4);

for (i=1:epocas) #se itera segun cierta cantidad de epocas
  
  for (j=1:rows(Xraw)/m) #cierta cantidad de minilotes por cada epoca
    
    ## Numero random para seleccion del mini lote
    num_random = randi(rows(Xraw)-m);
    
    X = Xraw(num_random:(num_random+m-1),:); #x tiene 2 features
    Y = Yraw(num_random:(num_random+m-1),:); #y de entrenamiento
    
    ## Forward prop
    y1a = l1a.forward(W1,X);  #se combinan datos y pesos
    y1b = l1b.forward(y1a);   #se pasa por funcion de activacion
    
    y2a = l2a.forward(W2,y1b);
    y2b = l2b.forward(y2a);
    
    y3a = l3a.forward(W3,y2b);
    y3b = l3b.forward(y3a);
    
    
    ## Backward prop
    y4 = l4.backward(y3b,Y); #gradiente de J con respecto a Y
      
    l3b.backward(y4);
    l3a.backward(l3b.gradient);
    
    l2b.backward(l3a.gradientX);
    l2a.backward(l2b.gradient);

    l1b.backward(l2a.gradientX); #a la capa 1 se le pasa el gradiente con respecto a X
    l1a.backward(l1b.gradient);

    ## Calculo de pesos
    W1 = W1 - alpha*mtm1.filtro(l1a.gradientW);
    W2 = W2 - alpha*mtm2.filtro(l2a.gradientW);
    W3 = W3 - alpha*mtm3.filtro(l3a.gradientW);
   
  endfor
  
  ## Calculo de error
  J = l4.error();
  Jacumulados = [Jacumulados; J];
  
  numEpocas = [numEpocas; i];
  
  disp(["Epoca: ", num2str(i),"/",num2str(epocas),"  J: ",num2str(J)]);
endfor

plot(numEpocas,Jacumulados);
xlabel("Epochs")
ylabel("Loss")
title("Loss vs Epochs with momentum")