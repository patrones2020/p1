function plot_loss(epochs,Jacumulados,color)
  plot(epochs,Jacumulados,color);
  xlabel("Epochs");
  ylabel("Loss");
  title("Loss vs Epochs");
  axis([0 inf 0 1]);
endfunction
