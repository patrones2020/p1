
function r=reludecision(x)
  r = x .* (x > 0); ##Los negativos dan -0
endfunction
