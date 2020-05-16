###Clase model

classdef model < handle
  properties
    alpha = 0.03; #Learning rate
    epochs = 2500;
    minilote = 100; ##Tamano del minilote
    clases = 5;
    neurons1 = 5;
    neurons2 = 3;
    dimensionX = 2; ##Cantidad de features o columnas del conjunto de datos
    ##Capas
    l1a = fullyConnectedBiased(); #combinacion
    l1b = tanHiperbolica();             #activacion
    l2a = fullyConnectedBiased(); #combinacion
    l2b = tanHiperbolica();
    l3a = fullyConnectedBiased(); #combinacion
    l3b = sigmoide(); #activacion
    
    l4 = MSE();
    
    #Pesos
    W1 = [];
    W2 = [];
    W3 = [];
  endproperties
  
  methods
    function s = model()
      s.init()
    endfunction
    
    function init(s)
      ##s.alpha = 0.03;
      ##s.epochs = 5000;
      ##s.minilote = 100;
      ##s.clases = 5;
      ##s.neurons1 = 5;
      ##s.neurons2 = 3;
      ##s.dimensionX = 2;
      s.W1 = rand(s.neurons1, s.dimensionX + 1);
      s.W2 = rand(s.neurons2, s.neurons1 + 1);
      s.W3 = rand(s.clases, s.neurons2 + 1);
    endfunction
    
    ##Funcion para guardar las matrices de pesos
    function save(s,file)
      W1 = s.W1;
      W2 = s.W2;
      W3 = s.W3;
      
      save(file,"W1","W2","W3");
    endfunction
    
    ##Funcion para cargar las matrices de pesos y parametros importantes
    function o = load(s,file)
      load(file);
      s.W1 = W1;
      s.W2 = W2;
      s.W3 = W3;
      
      s.dimensionX = columns(s.W1) - 1;
      s.neurons1 = rows(s.W1);
      s.neurons2 = rows(s.W2);
      s.clases = rows(s.W3);
    endfunction
    
    ##Funcion para entrenar los datos sin validacion
    function train(s,Xraw,Yraw)
      Jacumulados = [];
      numEpochs = [];
      
      for (i = 1:s.epochs)
        for (j = 1:rows(Xraw)/s.minilote)
          ## Seleccion de mini lote
          k = randperm(rows(Xraw)); #crea un vector con valores random de 1 a n, sin repetir
          X = Xraw(k(1:s.minilote),:); #toma una muestra de X, de tamaño = minilote
          Y = Yraw(k(1:s.minilote),:);
          
          ## Forward prop
          y1a = s.l1a.forward(s.W1,X);  #se combinan datos y pesos
          y1b = s.l1b.forward(y1a);   #se pasa por funcion de activacion
          
          y2a = s.l2a.forward(s.W2,y1b);
          y2b = s.l2b.forward(y2a);
          
          y3a = s.l3a.forward(s.W3,y2b);
          y3b = s.l3b.forward(y3a);
          
          
          ## Backward prop
          y4 = s.l4.backward(y3b,Y); #gradiente de J con respecto a Y
            
          s.l3b.backward(y4);
          s.l3a.backward(s.l3b.gradient);
          
          s.l2b.backward(s.l3a.gradientX);
          s.l2a.backward(s.l2b.gradient);
          
          s.l1b.backward(s.l2a.gradientX); #a la capa 1 se le pasa el gradiente con respecto a X
          s.l1a.backward(s.l1b.gradient);
          
          
          
          
          ## Calculo de pesos
          s.W1 = s.W1 - s.alpha*s.l1a.gradientW;
          s.W2 = s.W2 - s.alpha*s.l2a.gradientW;
          s.W3 = s.W3 - s.alpha*s.l3a.gradientW;
        endfor
        J = s.l4.error();
        Jacumulados = [Jacumulados;J];
        numEpochs = [numEpochs; i];
        
        disp(["Epoca: ", num2str(i),"/",num2str(s.epochs),"  J: ",num2str(J)]);
      endfor
      plot_loss(numEpochs,Jacumulados);
    endfunction
    
    ##Funcion para el test
    ##...
  endmethods
endclassdef
