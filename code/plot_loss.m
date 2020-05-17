function plot_loss(epochs,Jacumulados)
  plot(epochs,Jacumulados);
  xlabel("Epochs");
  ylabel("Loss");
  title("Loss vs Epochs");
  axis([0 inf 0 1]);
endfunction
