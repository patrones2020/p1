#!/usr/bin/octave-cli

## "Capa" de Entropía Cruzada
classdef cross_entropy < handle
  properties
    ## Salidas Y de la red (predicciones)
    inputsYpred = []
    ## Valores reales de Y (etiquetas)
    inputsY = []
    ## Error de salida J
    output = [];
    ## Gradiente de J con respecto a Y
    gradient=[];
  endproperties

  methods
    ## Constructor
    function s=cross_entropy()
      s.inputsYpred = [];
      s.inputsY = [];
      s.output = [];
      s.gradient=[];
    endfunction
    
    ## Se requiere hacer el calculo de gradiente antes de calcular J
    ## Porque es cuando se calcula el gradiente que se ingresan los datos

    ## Propagacion hacia atras calcula gradiente de J con respecto Y
    function y = backward(s,Ypred,Y)
      s.inputsYpred = Ypred;
      s.inputsY = Y;
      
      grad = Y;
      for(i = 1: columns(Y))
        for(j = 1: rows(Y))
          grad(j,i) = -(1/rows(Y) * (Y(j,i) / Ypred(j,i)));
        endfor
      endfor
      
      s.gradient = grad;
      y = s.gradient;
    endfunction
    
    #entropía cruzada de vectores columna
    function j = xent(s,y,ypred)
      j = y*log(ypred');
    endfunction  
    
    ## Calculo del error
    function y = error(s)
      a = 0;
      for(i = 1: rows(s.inputsY))
        a += s.xent(s.inputsY(i,:),s.inputsYpred(i,:));
      endfor
      s.output = -a / rows(s.inputsY);
      y = s.output;
      s.gradient = [];
    endfunction
    
  endmethods
  
endclassdef