## Momentum

classdef momentum < handle
  properties
    gradient = [];
    
    Vold = [];
    
    Vnew = [];
    
    beta = 0;
  endproperties
  
  methods
    function s = momentum()
      s.gradient = [];
      s.Vold = [];
      s.Vnew = [];
      s.beta = 0;
    endfunction
    
    function vt1 = filtro(s,gradientW,vt0,beta)
      s.gradient = gradientW;
      s.beta = beta;
      s.Vold = vt0;
      s.Vnew = beta .* vt0 + (1 - beta) .* gradientW;
      vt1 = s.Vnew;
    endfunction
  endmethods
endclassdef
