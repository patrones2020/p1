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
      for i = 1: colums(s.inputsY)
        sum += inputY(:,i)*(1/inputsYpred(:,i));
      endfor
      s.gradient = -sum; # en caso de error transponer el gradiente -sum'
      y = s.gradient;
    endfunction
    
    ## Calculo del error
    function y = error(s)
      sum = 0;
      for i = 1: colums(s.inputsY)
        sum += -inputY(:,i) * log(inputsYpred(:,i));
      endfor
      s.output = (1/columns(s.inputsY))*sum;
      y = s.output;
      s.gradient = [];
    endfunction
  endmethods
endclassdef