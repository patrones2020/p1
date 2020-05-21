#!/usr/bin/octave-cli

## "Capa" de Error
classdef loss < handle
  properties
    method;
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
    function lss = loss(method)
      lss.method = method;
      lss.inputsYpred = [];
      lss.inputsY = [];
      lss.output = [];
      lss.gradient=[];
    endfunction
    
    ## Se requiere hacer el calculo de gradiente antes de calcular J
    ## Porque es cuando se calcula el gradiente que se ingresan los datos

    ## Propagacion hacia atras calcula gradiente de J con respecto Y
    function y = backward(lss,Ypred,Y)
      lss.inputsYpred = Ypred;
      lss.inputsY = Y;
      switch(lss.method)
        case "mse"
          lss.gradient = (2/rows(Y))*(Ypred-Y);
        otherwise
          lss.gradient = (Y ./ Ypred)./-rows(Y);
      endswitch 
      y = lss.gradient;
    endfunction
    
    #entropía cruzada de vectores columna
    function j = xent(lss,y,ypred)
      j = y*log(ypred');
    endfunction  
    
    ## Calculo del error
    function y = error(lss)
  
      switch(lss.method)
        case "mse"
          lss.output = (1/rows(lss.inputsY))*sum((vecnorm((lss.inputsYpred-lss.inputsY)')').^2);
        otherwise
          a = 0;
          for(i = 1: rows(lss.inputsY))
            a += lss.xent(lss.inputsY(i,:),lss.inputsYpred(i,:));
          endfor
          lss.output = -a / rows(lss.inputsY);
      endswitch 
      y = lss.output;
      lss.gradient = [];
    endfunction
  endmethods
endclassdef