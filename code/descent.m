## Momentum

classdef descent < handle
  properties
    beta;
    beta2;
    s;
    v;
    epsilon = 0.01;
  endproperties
  
  methods
    function dsc = descent(beta, beta2)
      dsc.beta = beta;
      dsc.beta2 = beta2;
      dsc.v = []
      dsc.s = [];
    endfunction
    
    function v = pure(dsc, grad)
      v = grad;
    endfunction
    
    #en la primera iteracion deja no hay filtro
    function v = momentum(dsc, grad)
      if (isempty(dsc.v))
        dsc.v = grad;
      else 
        dsc.v = dsc.beta .* dsc.v + (1 - dsc.beta) .* grad;
      endif
      v = dsc.v;
    endfunction
    
    #en la primera iteracion deja no hay filtro
    function v = adam(dsc, grad)
      if (isempty(dsc.s)) 
        dsc.s = grad .^ 2;
      else 
        dsc.s = dsc.beta2 .* dsc.s + (1 - dsc.beta2) .* grad .^ 2;
      endif
      v = dsc.momentum(grad) ./ sqrt(dsc.s + dsc.epsilon);
    endfunction
  endmethods
endclassdef