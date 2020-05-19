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
    l1b = sigmoide();                 #activacion
    l2a = fullyConnectedBiased();
    l2b = sigmoide();
    l3a = fullyConnectedBiased();
    l3b = sigmoide();
    l4a = fullyConnectedBiased();
    l4b = sigmoide();
    
    lf = MSE(); #layer final, para calculo de error de entrenamiento
    lfVal = MSE(); #layer final, para calculo de error de validacion
    
    opt = "pure"; #metodo de optimizacion
    
    #Pesos
    W1 = [];
    W2 = [];
    W3 = [];
    W4 = [];
    
    #descenso
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
    function train(s,Xraw,Yraw,Xval,Yval)
      Jacumulados = [];
      JacumuladosVal = [];
      numEpochs = [];
      for (i = 1:s.epochs)
        for (j = 1:rows(Xraw)/s.minilote)
          ## Seleccion de mini lote
          k = randperm(rows(Xraw)); #crea un vector con valores random de 1 a n, sin repetir
          X = Xraw(k(1:s.minilote),:); #toma una muestra de X, de tamaño = minilote
          Y = Yraw(k(1:s.minilote),:);
          
          ## Forward prop Train
          y4b = forward_prop(s,X);
          
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
          if (strcmp(s.opt,"momentum"))
            s.W1 = s.W1 - s.alpha*s.dsc1.momentum(s.l1a.gradientW);
            s.W2 = s.W2 - s.alpha*s.dsc2.momentum(s.l2a.gradientW);
            s.W3 = s.W3 - s.alpha*s.dsc3.momentum(s.l3a.gradientW);
            s.W4 = s.W4 - s.alpha*s.dsc4.momentum(s.l4a.gradientW);
          elseif (strcmp(s.opt,"adam"))
            s.W1 = s.W1 - s.alpha*s.dsc1.adam(s.l1a.gradientW);
            s.W2 = s.W2 - s.alpha*s.dsc2.adam(s.l2a.gradientW);
            s.W3 = s.W3 - s.alpha*s.dsc3.adam(s.l3a.gradientW);
            s.W4 = s.W4 - s.alpha*s.dsc4.adam(s.l4a.gradientW);
          else
            s.W1 = s.W1 - s.alpha*s.dsc1.pure(s.l1a.gradientW);
            s.W2 = s.W2 - s.alpha*s.dsc2.pure(s.l2a.gradientW);
            s.W3 = s.W3 - s.alpha*s.dsc3.pure(s.l3a.gradientW);
            s.W4 = s.W4 - s.alpha*s.dsc4.pure(s.l4a.gradientW);
          endif
          
          ##Forward prop Validation
          y4b = forward_prop(s,Xval);
          
          yfval = s.lfVal.backward(y4b,Yval);
          
        endfor
        J = s.lf.error();
        Jacumulados = [Jacumulados;J];
        JVal = s.lfVal.error();
        JacumuladosVal = [JacumuladosVal;JVal]; 
        numEpochs = [numEpochs; i];
        
        disp(["Epoca: ", num2str(i),"/",num2str(s.epochs),"  J: ",num2str(J)]);
      endfor
      figure
      hold on
      plot_loss(numEpochs,Jacumulados,"y"); ##Grafica el error vs epocas
      plot_loss(numEpochs,JacumuladosVal,"b");
      Legend=cell(2,1);
      Legend{1}=strcat('Train');
      Legend{2}=strcat('Validation');
      legend(Legend);
      
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
      y4b = forward_prop(s,Pixels);
      
      ## Plot de datos y prediccion
      plot_colors(y4b,s.clases,Xraw,Yraw);
      
    endfunction

  endmethods
endclassdef