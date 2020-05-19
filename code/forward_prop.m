## Forward propagation con 4 capas

function y4b = forward_prop(s,X)
  y1a = s.l1a.forward(s.W1,X);  #se combinan datos y pesos
  y1b = s.l1b.forward(y1a);   #se pasa por funcion de activacion
  
  y2a = s.l2a.forward(s.W2,y1b);
  y2b = s.l2b.forward(y2a);
  
  y3a = s.l3a.forward(s.W3,y2b);
  y3b = s.l3b.forward(y3a);
  
  y3a = s.l3a.forward(s.W3,y2b);
  y3b = s.l3b.forward(y3a);
  
  y4a = s.l4a.forward(s.W4,y3b);
  y4b = s.l4b.forward(y4a);
  
endfunction
