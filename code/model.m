###Clase model

classdef model < handle
  properties
    alpha = 0.03; #Learning rate
    epochs = 2500;
    minilote = 100; ##Tamano del minilote
    clases = 5;
    neurons1 = 5;
    neurons2 = 5;
    neurons3 = 5;
    dimensionX = 2; ##Cantidad de features o columnas del conjunto de datos
    ##Capas
    l1a = fullyConnectedBiased(); #combinacion
    l1b = relu();                 #activacion
    l2a = fullyConnectedBiased();
    l2b = relu();
    l3a = fullyConnectedBiased();
    l3b = relu();
    l4a = fullyConnectedBiased();
    l4b = sigmoide();
    
    lf = MSE(); #layer final, para calculo de error
    
    #Pesos
    W1 = [];
    W2 = [];
    W3 = [];
    W4 = [];
    
    #descenso
    beta = 0.9;
    beta2 = 0.99;
    dsc1 = descent(0.9, 0.99);
    dsc2 = descent(0.9, 0.99);
    dsc3 = descent(0.9, 0.99);
    dsc4 = descent(0.9, 0.99);
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
      s.W3 = rand(s.neurons3, s.neurons2 + 1);
      s.W4 = rand(s.clases, s.neurons3 + 1);
    endfunction
    
    ##Funcion para guardar las matrices de pesos
    function save(s,file)
      W1 = s.W1;
      W2 = s.W2;
      W3 = s.W3;
      W4 = s.W4;
      
      save(file,"W1","W2","W3","W4");
    endfunction
    
    ##Funcion para cargar las matrices de pesos y parametros importantes
    function o = load(s,file)
      load(file);
      s.W1 = W1;
      s.W2 = W2;
      s.W3 = W3;
      s.W4 = W4;
      
      s.dimensionX = columns(s.W1) - 1;
      s.neurons1 = rows(s.W1);
      s.neurons2 = rows(s.W2);
      s.neurons3 = rows(s.W3);
      s.clases = rows(s.W4);
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
          
          y3a = s.l3a.forward(s.W3,y2b);
          y3b = s.l3b.forward(y3a);
          
          y4a = s.l4a.forward(s.W4,y3b);
          y4b = s.l4b.forward(y4a);
          
          ## Backward prop
          yf = s.lf.backward(y4b,Y); #gradiente de J con respecto a Y
          
          s.l4b.backward(yf);
          s.l4a.backward(s.l4b.gradient);  
        
          s.l3b.backward(s.l4a.gradientX);
          s.l3a.backward(s.l3b.gradient);
          
          s.l2b.backward(s.l3a.gradientX);
          s.l2a.backward(s.l2b.gradient);
          
          s.l1b.backward(s.l2a.gradientX); #a la capa 1 se le pasa el gradiente con respecto a X
          s.l1a.backward(s.l1b.gradient);
          
          ## Calculo de pesos
          s.W1 = s.W1 - s.alpha*s.dsc1.pure(s.l1a.gradientW);
          s.W2 = s.W2 - s.alpha*s.dsc2.pure(s.l2a.gradientW);
          s.W3 = s.W3 - s.alpha*s.dsc3.pure(s.l3a.gradientW);
          s.W4 = s.W4 - s.alpha*s.dsc4.pure(s.l4a.gradientW);
        endfor
        J = s.lf.error();
        Jacumulados = [Jacumulados;J];
        numEpochs = [numEpochs; i];
        
        disp(["Epoca: ", num2str(i),"/",num2str(s.epochs),"  J: ",num2str(J)]);
      endfor
      plot_loss(numEpochs,Jacumulados); ##Grafica el error vs epocas
    endfunction
    
    ## Funcion para predecir y plotear
    ## los resultados con la red ya entrenada
    function predict(s,Xraw,Yraw)
      
      ## Se realiza grid de 512x512
      ## Y se predice la salida para cada pixel
      x = linspace(-1,1,512);
      [GX,GY] = meshgrid(x,x);
      Pixels = [GX(:) GY(:)];

      ## Forward prop
      y1a = s.l1a.forward(s.W1,Pixels);  #se combinan datos y pesos
      y1b = s.l1b.forward(y1a);   #se pasa por funcion de activacion
      
      y2a = s.l2a.forward(s.W2,y1b);
      y2b = s.l2b.forward(y2a);
      
      y3a = s.l3a.forward(s.W3,y2b);
      y3b = s.l3b.forward(y3a);
      
      y3a = s.l3a.forward(s.W3,y2b);
      y3b = s.l3b.forward(y3a);
      
      y4a = s.l4a.forward(s.W4,y3b);
      y4b = s.l4b.forward(y4a);
      
      ## Plot de datos
      plot_data(Xraw,Yraw);
      
      ## Plot de prediccion
      plot_colors(y4b,s.clases);
      
    endfunction
    
    ## Funcion para crear la matriz de confusióń
    ## Recibe X y Y de un se 
    function confusion(s,Xraw,Yraw)

      ## Forward prop
      y1a = s.l1a.forward(s.W1,Xraw);  #se combinan datos y pesos
      y1b = s.l1b.forward(y1a);   #se pasa por funcion de activacion
      
      y2a = s.l2a.forward(s.W2,y1b);
      y2b = s.l2b.forward(y2a);
      
      y3a = s.l3a.forward(s.W3,y2b);
      y3b = s.l3b.forward(y3a);
      
      y3a = s.l3a.forward(s.W3,y2b);
      y3b = s.l3b.forward(y3a);
      
      y4a = s.l4a.forward(s.W4,y3b);
      y4b = s.l4b.forward(y4a);
      
      Ypred = y4a;

      ## Calculo de la matriz de prediccion
      
      conf = zeros(columns(Yraw), columns(Yraw)); #inicializa matriz de confusion
      
      for(i = 1:rows(Yraw))
        [max_value index] = max(Ypred(i,:)); #obtiene el indice el valor mas grande del la fila de Ypred
        conf(find(Yraw(i,:),1),index)++; #incrementa el 
      endfor
      
      ## Calculo de la exhaustividad
     
      recall = zeros(1,columns(Yraw)); #inicializa arreglo de exhaustividad
      
      for(i = 1:rows(conf))
        recall(i) = conf(i,i) / sum(conf(i,:)); # True
      endfor
      
      ## Calculo de la precision
      
      precision = zeros(1,columns(Yraw));
      
      for(i = 1:columns(conf))
        precision(i) = conf(i,i) / sum(conf(:,i)); 
      endfor
      
      ## Calculo de f1
      f1 = zeros(1,columns(Yraw));
      
      for(i = 1:columns(conf))
        tp = conf(i,i);
        fp = sum(conf(:,i)) - tp;
        fn = sum(conf(i,:)) - tp;
        f1(i) = (2 * tp) / ( 2 * tp + fp + fn); 
      endfor
      
      ##Display de metricas 
      
      #disp(Yraw);
      #disp(Ypred);
      
      disp("matriz de confusion");
      disp(conf);
      
      disp("exhaustividad");
      disp(recall);
      
      disp("precision");
      disp(precision);
      
      disp("f1");
      disp(f1);
      
      
    endfunction
    
    
    ##Funcion para el test
    ##...
    
    
  endmethods
endclassdef