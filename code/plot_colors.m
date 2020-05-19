## Plot de resultados de la prediccion

## FZ es la matriz de valores predichos
function plot_colors(FZ,numClasses,X,Y)
  
  ## Solo importa la dimension de GX
  ## En este caso se obtiene de FZ
  GX = magic(sqrt(size(FZ)(1)));
  
  ## Normalizacion de la salida
  FZ = FZ./sum(FZ,2);
  FZ = FZ';
  
  ## Color map
  cmap = [1  ,0  ,0  ;
          0  ,0.7,0  ;
          0  ,0  ,0.8; 
          1  ,0  ,1  ;
          0  ,0.7,0.7;
          0.8,0.6,0.0; 
          0.8,0.5,0.2;
          0.2,0.5,0.3;
          0.6,0.3,0.8;
          0.6,0.1,0.4;
          0.6,0.8,0.3;
          0.1,0.4,0.6;
          0.5,0.5,0.5];
  
  ## Plot de datos
  figure("name","Datos");
  plot_data(X,Y);
  title("Datos");
  
  ## Plot segun probabilidades
  ccmap = cmap(1:numClasses,:);
  cwimg = ccmap'*FZ;
  redChnl   = reshape(cwimg(1,:),size(GX));
  greenChnl = reshape(cwimg(2,:),size(GX));
  blueChnl  = reshape(cwimg(3,:),size(GX));
   
  mixed = flip(cat(3,redChnl,greenChnl,blueChnl),1);
  figure("name","Clases segun probabilidades");;
  
  #resize de la figura
  him = imshow(mixed, []);
  set(him, 'XData', [-1, 1], 'YData', [-1,1]);
  axis([-1 1 -1 1]);
  
  axis on;
  title("Clases segun probabilidades");
  
  
  ## Plot de la clase ganadora
  [maxprob,maxk] = max(FZ);
    
  winner=flip(uint8(reshape(maxk,size(GX))),1);
          
  wimg=ind2rgb(winner,[0,0,0;cmap]);
  figure("name","Clases ganadoras");
  
  #resize de la figura
  him = imshow(wimg, []);
  set(him, 'XData', [-1, 1], 'YData', [-1,1]);
  axis([-1 1 -1 1]);
  
  axis on;
  title("Clases ganadoras");
  
  
  ## Plot de datos sobre los colores
  figure("name","Datos y predicción");
  him = imshow(wimg, []);
  set(him, 'XData', [-1, 1], 'YData', [-1,1]);
  axis([-1 1 -1 1]);
  hold on;
  plot_data(X,Y,"brighter");
  title("Datos y predicción");

endfunction
