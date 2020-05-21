###Clase model

classdef model1 < handle
  properties
    clases;
    alpha; #Learning rate
    minilote; ##Tamano del minilote 
 #   clases; # = 5
    W; #arreglo de matrices de pesos
    comb; #arreglo con las capas de combinacion
    act;  #arreglo con las capas de activacion
    loss; #capa de error;
    dsc;
    lfVal = MSE(); #layer final, para calculo de error de validacion
    epochs = 3000;
  endproperties
  
  methods
    function mdl = model1(clases, alpha, minilote, activation,layerSize, lossMethod, dscMethod, beta, beta2)
      mdl.clases = clases;
      mdl.alpha = alpha;
      mdl.minilote = minilote;
      mdl.act = activation;
      mdl.loss = loss(lossMethod);
      
      # Metodos de descenso
      mdl.dsc = cell(1,columns(mdl.act));
      for i = 1:columns(mdl.act);
        mdl.dsc{1,i} = descent(dscMethod,beta,beta2);
      endfor
      
      #capa de combinacion
      mdl.comb = cell(1,columns(mdl.act));
      for i = 1:columns(mdl.act);
        mdl.comb{1,i} = fullyConnectedBiased();
      endfor
      
      #pesos de funcion
      mdl.W = cell(1,columns(mdl.act));
      mdl.W{1,1} = rand(layerSize(1), 2 +1); #input: columnas del conjunto de datos + sesgo
      for i = 2:columns(mdl.act)
        mdl.W{1,i} = rand(layerSize(i), layerSize(i-1) + 1);
      endfor
      
    endfunction
    
#{
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

#}
    
    ##Funcion para entrenar los datos sin validacion
    function train1(mdl,Xraw,Yraw,Xval,Yval)
      Jacumulados = [];
      JacumuladosVal = [];
      numEpochs = [];
      for (i = 1:mdl.epochs)
        for (j = 1:rows(Xraw)/mdl.minilote)
          ## Seleccion de mini lote
          k = randperm(rows(Xraw)); #crea un vector con valores random de 1 a n, sin repetir
          X = Xraw(k(1:mdl.minilote),:); #toma una muestra de X, de tamaño = minilote
          Y = Yraw(k(1:mdl.minilote),:);
          
          data = X; #used to pass data between layers
          
          #Forward Propagation
          for l = 1:columns(mdl.act)
            data = mdl.comb{1,l}.forward(mdl.W{1,l},data);
            data = mdl.act{1,l}.forward(data);
          endfor
             
          #Backward Propagation
          data = mdl.loss.backward(data,Y); # data = Ypred
          
          for l = columns(mdl.act):-1:1
            #act
            mdl.act{1,l}.backward(data);
            data = mdl.act{1,l}.gradient;
            
            #comb
            mdl.comb{1,l}.backward(data);
            data = mdl.comb{1,l}.gradientX;
          endfor
         
          ## Calculo de pesos
          for l = 1:columns(mdl.W)
            mdl.W{1,l} = mdl.W{1,l} - mdl.alpha * mdl.dsc{1,l}.filter(mdl.comb{1,l}.gradientW);
          endfor
          
          
          #{
          ##Forward prop Validation
          y4b = forward_prop(s,Xval);
          
          yfval = s.lfVal.backward(y4b,Yval);
          #}
        endfor
        J = mdl.loss.error();
        Jacumulados = [Jacumulados;J];
        #JVal = s.lfVal.error();
        #JacumuladosVal = [JacumuladosVal;JVal]; 
        numEpochs = [numEpochs; i];
        
        disp(["Epoca: ", num2str(i),"/",num2str(mdl.epochs),"  J: ",num2str(J)]);
      endfor
      figure
      hold on
      plot_loss(numEpochs,Jacumulados,"r"); ##Grafica el error vs epocas
     # plot_loss(numEpochs,JacumuladosVal,"b");
      #Legend=cell(2,1);
      #Legend{1}=strcat('Train');
     # Legend{2}=strcat('Validation');
      #legend(Legend);
      
    endfunction
    
    ## Funcion para predecir y plotear
    ## los resultados con la red ya entrenada
    function predict1(mdl,Xraw,Yraw)
      
      ## Se realiza grid de 512x512
      ## Y se predice la salida para cada pixel
      x = linspace(-1,1,512);
      [GX,GY] = meshgrid(x,x);

      data = [GX(:) GY(:)];
      
      #Forward Propagation
      for l = 1:columns(mdl.act)
        data = mdl.comb{1,l}.forward(mdl.W{1,l},data);
        data = mdl.act{1,l}.forward(data);
      endfor
      
      ## Plot de datos y prediccion
      plot_colors(data,mdl.clases,Xraw,Yraw); 
    endfunction
    
        ## Funcion para crear la matriz de confusióń
    ## Recibe X y Y de un se 
    function confusion1(mdl,Xraw,Yraw)

      data = Xraw;
      
      #Forward Propagation
      for l = 1:columns(mdl.act)
        data = mdl.comb{1,l}.forward(mdl.W{1,l},data);
        data = mdl.act{1,l}.forward(data);
      endfor
      
      Ypred = data;

      ## Calculo de la matriz de prediccion
      
      conf = zeros(columns(Yraw), columns(Yraw)); #inicializa matriz de confusion
      
      for(i = 1:rows(Yraw))
        [max_value index] = max(Ypred(i,:)); #obtiene el indice el valor mas grande del la fila de Ypred
        conf(find(Yraw(i,:),1),index)++; #incrementa el 
      endfor
      
      ## Calculo de la exhaustividad
     
      recall = zeros(1,columns(Yraw)); #inicializa arreglo de exhaustividad
      
      for(i = 1:rows(conf))
        posCond = sum(conf(i,:));
        if posCond == 0
          recall(i) = 0
        else  
          recall(i) = conf(i,i) / posCond; # True
        endif
      endfor
      
      ## Calculo de la precision
      
      precision = zeros(1,columns(Yraw));
      
      for(i = 1:columns(conf))
        predPosCond = sum(conf(:,i));
        if predPosCond == 0
          precision(i) = 0;
        else  
          precision(i) = conf(i,i) / predPosCond;
        endif
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
    
  endmethods
endclassdef