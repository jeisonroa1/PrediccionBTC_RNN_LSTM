%% Implementación CNN para la predicción del precio del Bitcoin
%% Jeison Roa 2018
clear all; close all; clc;
load ('data.mat');
T1 = 8;                                                % Periodo Rápido
T11 = 15;                                              % Periodos MACD Lento
T2 = 40;                                               % Periodo Lento
%% Reducción del Batch (Nota: Cambiar para efectos de una normalización adecuada)
n = 60;    % Default: 1  
m = 1050; % Default: 1604 Usando 100% de los datos
precio = precio(n:m);
minimo = minimo(n:m);
maximo = maximo(n:m);
volumen = volumen(n:m);
%% Señal de Volumen (Elimina valores erroneos)

for i =1:length(volumen)
    if isnan(volumen(i))== 1
        volumen(i)= 0;
    end
end
%% Señales Osciladores Estocásticos
% Oscilador Rapido T1
for i = 1:(length(precio)-T1)
OscEr(i+T1) = (precio(i+T1)- min(precio(i:i+T1)))/(max(precio(i:i+T1))-min(precio(i:i+T1)));
end
% Oscilador Lento T2

for i = 1:(length(precio)-T2)
OscEl(i+T2) = (precio(i+T2)- min(precio(i:i+T2)))/(max(precio(i:i+T2))-min(precio(i:i+T2)));
end

for i =1:length(OscEl)
    if isnan(OscEl(i))== 1
        OscEl(i)= 0;
    end
    if isnan(OscEr(i))== 1
        OscEr(i)= 0;
    end
end

%% Señales MACD
% Señal MACD Rapido
EMA11 = tsmovavg(precio,'e',T1);
EMA12 = tsmovavg(precio,'e',T2);
dif1 = EMA11-EMA12;
MACDr = tsmovavg(dif1(40:length(dif1)),'e',9);
MACDr = [zeros(1,39) MACDr];
% Señal MACD lento

EMA21 = tsmovavg(precio,'e',T11);
EMA22 = tsmovavg(precio,'e',T2);
dif2 = EMA21-EMA22;
MACDl = tsmovavg(dif2(40:length(dif2)),'e',9);
MACDl = [zeros(1,39) MACDl];

for i =1:length(MACDl)
    if isnan(MACDl(i))== 1
        MACDl(i)= 0;
    end
    if isnan(MACDr(i))== 1
        MACDr(i)= 0;
    end
end

%% Señal Indicador RSI
for i = 1:length(precio)-1
    
    if precio(i+1) > precio(i)
        u(i+1) = precio(i+1)-precio(i);
        d(i+1) = 0;
    else
        u(i+1) = 0;
        d(i+1) = precio(i)-precio(i+1);
        if precio(i+1) == precio(i)
            u(i+1) = 0;
            d(i+1) = 0;
        end
    end

end
RS =  tsmovavg(u,'e',T1)./tsmovavg(d,'e',T1);
RSI = 100 - 100.*(1./(1+RS));
for i =1:length(RSI)
    if isinf(RSI(i))== 1
        RSI(i)= 100;
    end
    if isnan(RSI(i))== 1
        RSI(i)= 0;
    end
end

%% Señal A/D Oscilator

for i = 1:length(precio)
ADosc(i) = (((precio(i)-minimo(i))-(maximo(i)-precio(i)))/((maximo(i)-precio(i))))*volumen(i);
end
for i =1:length(ADosc)
    if isinf(ADosc(i))== 1
        ADosc(i)= 260;
    end
    if isnan(ADosc(i))== 1
        ADosc(i)= 0;
    end
end

%% Señal ROC

for i = 1:length(precio)-T1
ROC(i+T1) = ((precio(i+T1)-precio(i))/precio(i))*100;
end
for i =1:length(ROC)
    if isinf(ROC(i))== 1
        ROC(i)= 100;
    end
    if isnan(ROC(i))== 1
        ROC(i)= 0;
    end
end

%% Normalización de las entradas

precioN = (precio - min(precio))/(max(precio)-min(precio));
preciopasadoN = [0 precioN(1:length(precioN)-1)];
volumenN = (volumen - min(volumen))/(max(volumen)-min(volumen));
MACDrN = (MACDr - min(MACDr))/(max(MACDr)-min(MACDr));
MACDlN = (MACDl - min(MACDl))/(max(MACDl)-min(MACDl));
ROCN = (ROC - min(ROC))/(max(ROC)-min(ROC));
RSIN = (RSI - min(RSI))/(max(RSI)-min(RSI));
OscErN = OscEr;
OscElN = OscEl;
ADoscN = (ADosc - min(ADosc))/(max(ADosc)-min(ADosc));
minimoN = (minimo - min(minimo))/(max(minimo)-min(minimo));
maximoN = (maximo - min(maximo))/(max(maximo)-min(maximo));

%% Generación matriz de entrenamiento  y validación
% a = inicio de datos de entrenamiento. b+1 = inicio datos de validación.
% Verificar Linea 9. Debe tener coherencia con el tamaño del batch (m-n).
p = round((m-n)*0.7);
a = 1;     % Default: 1
b = p;     % Default: 1200  si en la linea 9 se aprovecha el 100%
a = a+40;  % Se descartan los primeros 40 datos dado que son cero (MACD)
%Entrenamiento
entradas = [precioN(a:b) ; preciopasadoN(a:b) ; volumenN(a:b) ; MACDrN(a:b) ; MACDlN(a:b); ROCN(a:b); RSIN(a:b); OscErN(a:b); OscElN(a:b) ; ADoscN(a:b) ; minimoN(a:b) ; maximoN(a:b)];
salidas = [precioN(a+1:b) precioN(b)]';

%Validación
entradasVal = [precioN(b+1:length(precioN)) ; preciopasadoN(b+1:length(precioN)) ; volumenN(b+1:length(precioN)) ; MACDrN(b+1:length(precioN)) ; MACDlN(b+1:length(precioN)); ROCN(b+1:length(precioN)); RSIN(b+1:length(precioN)); OscErN(b+1:length(precioN)); OscElN(b+1:length(precioN)) ; ADoscN(b+1:length(precioN)) ; minimoN(b+1:length(precioN)) ; maximoN(b+1:length(precioN))];
salidasVal = [precioN(b+2:length(precioN)) precioN(length(precioN))]';

%% Entrenamiento Red neuronal Convolucional
% Definición de la arquitectura
input = sequenceInputLayer(12); % 12 valor fijo.
learnable1 = fullyConnectedLayer(5000); % lstmLayer, bilstmLayer, fullyConnectedLayer(5000). convolution2dLayer
learnable2 = bilstmLayer(100);
relu = clippedReluLayer(300); % reluLayer, leakyReluLayer, clippedReluLayer(30).
fc1 = fullyConnectedLayer(50); % 5 , 10, 15, 20.
fc2 = fullyConnectedLayer(1); 
out = regressionLayer;
capas = [
    input
    learnable1
    learnable2
    relu
    fc1
    fc2
    out];
% Definición opciones de entrenamiento
opciones = trainingOptions('rmsprop'); %% 'sgdm'| 'rmsprop' | 'adam'
% Entrenamiento
Net = trainNetwork(entradas,salidas',capas,opciones);

%% Validación
[Y] = predict(Net,entradasVal);   % Cambiar por entradas para validar con datos Ent.
acumulado = 0 ;
for i=1:length(Y)
    acumulado = acumulado + (salidasVal(i)-Y(i))^2;
end
RMSE = acumulado / length(Y)

% Calcula probabilidad de acierto de tendencia.
aciertos=0;
for i=1:length(Y)-1
    if entradasVal(1,i+1)> entradasVal(1,i)
        if Y(i)> entradasVal(1,i)
            aciertos = aciertos+1;
        end
    else
        if Y(i)< entradasVal(1,i)
            aciertos = aciertos+1;
        end
    end
end
ProbTendencia = (aciertos/(length(Y)-1))*100

%% Figuras (Señales Ent+val+Predicción )
% figure;
% plot (precioN(a:length(precioN)));
% hold on;
% plot ([zeros(1,length(entradas(1,:))) entradasVal(1,:) ],'r');
% hold on;
% plot ([zeros(1,length(entradas(1,:))) Y ],'g');
% set (gca,'fontsize',12); 
% title ('Señales de Normalizadas (Entrenamiento + Validación + Predicción)');
% xlabel ('Dias');
% ylabel ('Precio');
% legend('Entrenamiento', 'Validación', 'Predicción');

%% Zoom Validación
figure
plot(salidasVal,'r');   % Cambiar por salidas para validar con datos Ent
hold on;                % Si se grafica salidas, ignorar figura 1.
plot (Y)
set (gca,'fontsize',12); 
title ('Señal de Validación Vs Predicción');
xlabel ('Dias');
ylabel ('Precio');
legend('Validación','Predicción');
