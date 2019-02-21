clc;
clear all;

%Data = load('d_hat.dat');
Data = rand(1000,10);
N = size(Data,1);
x0 = ones(N,1);
X = [x0 Data];
w = rand(size(X,2),1);
eta = 0.01;
epsilon = 0.0005;


y = rand(N,1);

RMSE = 999;
count = 0;
while RMSE >= epsilon

    count = count +1;
    %Initial y_hat
    y_hat = w'*X';
    y_hat = y_hat';
    
    %Activation function (Sigmoid)
    y_pred = 1/(1+exp(-y_hat'));

    %Calculate initial errors, e_i's
    e = (y - y_hat);
   
    for j=1:size(w,1)
        %calculate delta_w_j
        sum_e = e'*X(:,j);
        delta_w(j) = -1/N*eta*sum_e;
        %Update w_i's
        w_new(j) = w(j) - delta_w(j);
        %Calculate RMSE of w_i's
        e_w(j) = (w_new(j) - w(j));
        %Swap old and new values for w_j's
        w(j) = w_new(j);
    end
    RMSE = sqrt(e_w*e_w');    
    RMSE_out(count) = RMSE;   
end
w
plot(RMSE_out)
    
    
    
    
    
    
    
    
    
    
    

 















