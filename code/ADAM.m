## Momentum

classdef ADAM < handle
  properties
    momentum;
    beta2;
    sOld;
    epsilon = 0.01;
  endproperties
  
  methods
    function s = ADAM(beta, beta2)
      s.momentum = momentum(beta);
      s.beta2 = beta2;
      s.sOld = [];
    endfunction
    
    #en la primera iteracion deja no hay filtro
    function v = filtro(s, grad)
      if (isempty(s.sOld)) 
        s.sOld = grad .^ 2;
      else 
        s.sOld = s.beta2 .* s.sOld + (1 - s.beta2) .* grad .^ 2;
      endif
      v = s.momentum.filtro(grad) ./ sqrt(s.sOld + s.epsilon);
    endfunction
  endmethods
endclassdef