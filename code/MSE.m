#!/usr/bin/octave-cli

## "Capa" de Error cuadratico medio (MSE)
classdef MSE < handle
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
    function s=MSE()
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
      s.gradient = (2/rows(Y))*(Ypred-Y);
      y = s.gradient;
    endfunction
    
    ## Calculo del error
    function y = error(s)
      s.output = (1/rows(s.inputsY))*sum((vecnorm((s.inputsYpred-s.inputsY)')').^2);
      y = s.output;
      s.gradient = [];
    endfunction
  endmethods
endclassdef