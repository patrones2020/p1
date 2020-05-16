#!/usr/bin/octave-cli

## "Capa" tangente hiperbolica
classdef tanHiperbolica < handle
  properties    
    ## Resultados despues de forward prop
    outputs=[];
    ## Resultados despues de backward prop
    gradient=[];
  endproperties

  methods
    ## Constructor ejecuta un forward si se le pasan datos
    function s=tanHiperbolica()
      s.outputs=[];
      s.gradient=[];
    endfunction

    ## Propagacion hacia adelante
    function y = forward(s,a)
      s.outputs = tanh(a);
      y = s.outputs;
      s.gradient = [];
    endfunction

    ## Propagacion hacia atras recibe dL/ds de siguientes nodos
    function backward(s,dLds)
      if (size(dLds)!=size(s.outputs))
        error("backward de tanh no compatible con forward previo");
      endif
      localGrad = 4./(exp(s.outputs)+exp(-s.outputs)).^2;
      s.gradient = localGrad.*dLds;
    endfunction
  endmethods
endclassdef