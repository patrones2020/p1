## Momentum

classdef momentum < handle
  properties
    beta = 0;
    vOld = [];
  endproperties
  
  methods
    function s = momentum(beta)
      s.beta = 0;
      s.vOld = [];
    endfunction
    
    #en la primera iteracion deja no hay filtro
    function v = filtro(s, gradientW)
      if (isempty(s.vOld))
        s.vOld = gradientW;
      else 
        s.vOld = s.beta .* s.vOld + (1 - s.beta) .* gradientW;
      endif
      v = s.vOld;
    endfunction
  endmethods
endclassdef
