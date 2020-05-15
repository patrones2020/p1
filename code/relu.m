##Funcion de activacion ReLU

##La salida sera el vector de entrada si este es mayor que 0
##Se recuerda la compuerta max

classdef relu < handle
  properties    
    ## Resultados despues de la propagacion hacia adelante
    outputs=[];
    ## Resultados despues de la propagacion hacia atras
    gradient=[];
  endproperties

  methods
    ## Constructor ejecuta un forward si se le pasan datos
    function s=relu()
      s.outputs=[];
      s.gradient=[];
    endfunction

    ## Propagacion hacia adelante
    function y=forward(s,a)
      s.outputs = reludecision(a);
      y=s.outputs;
      s.gradient = [];
    endfunction

    ## Propagacion hacia atras recibe dL/ds de siguientes nodos
    function backward(s,dLds)
      if (size(dLds)!=size(s.outputs))
        error("backward de relu no compatible con forward previo");
      endif
      localGrad = relugrad(s.outputs);
      s.gradient = localGrad.*dLds;
    endfunction
  endmethods
endclassdef
