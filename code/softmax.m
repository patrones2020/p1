#!/usr/bin/octave-cli

## "Capa" sigmoide, que aplica la función logística
classdef softmax < handle
  properties    
    ## Resultados después de la propagación hacia adelante
    outputs=[];
    ## Resultados después de la propagación hacia atrás
    gradient=[];
  endproperties

  methods
    ## Constructor ejecuta un forward si se le pasan datos
    function s=softmax()
      s.outputs=[];
      s.gradient=[];
    endfunction

    ## Propagación hacia adelante
    function y=forward(s,a)
      s.outputs = exp(a) ./ sum(exp(a),2);
      y=s.outputs;
      s.gradient = [];
    endfunction

    ## Propagación hacia atrás recibe dL/ds de siguientes nodos
    function backward(s,dLds)
      if (size(dLds)!=size(s.outputs))
        error("backward de sigmoide no compatible con forward previo");
      endif
      s.gradient = s.outputs .* dLds - (((s.outputs .* dLds) * ones(columns(s.outputs),1)) * ones(columns(s.outputs),1)') .* s.outputs;
    endfunction
  endmethods
endclassdef