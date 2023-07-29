%% LSTM/midterm/saman faridsoltani
clc;
close all;
clear;
%% Read data 
X=xlsread('DATA');
%% normalize
% x=zeros(size(X));
% for i=1:3
% x(:,i) = Normal_func(X(:,i),0,+1);
% end
for k=1:3
n = size(X(:,k),1);
m = size(X(:,k),2);
% input_num = m-1;

data_min = min(X(:,k));
data_max = max(X(:,k));

for i = 1:n
        X(i, k) = (X(i, k) - data_min) / (data_max - data_min);
%         data;
end
end
x=X;
%%  Making Data
 N = size(x(:,1),1);
    D=5;
  L=0;;
[Data1,Target1]=Make_Data(x(:,1),N,D,L);
 N = size(x(:,2),1);
    D=5;
  L=0;
[Data2,Target2]=Make_Data(x(:,2),N,D,L);
 N = size(x(:,3),1);
    D=20;
  L=0;
[Data3,Target3]=Make_Data(x(:,3),N,D,L);
DATA=zeros(size(Data1,1),3*D);
DATA=[Data1 Data2 Data3];
% plot(DATA)
TARGET=zeros(size(Data1,1),3);
TARGET=[Target1 Target2 Target3];
DATASET=[DATA TARGET];
%% parameters
n0_neurons=size(DATA,2);
n1_neurons=32;
n2_neurons=16;
l1_neurons=3;
lowerb=-1;
upperb=1;
epochs_p=1000;
eta_p=0.1;
%% Initialize Values
%% first LSTM
Wxi_1=unifrnd(lowerb,upperb,[n1_neurons n0_neurons+1]);
Whi_1=unifrnd(lowerb,upperb,[n1_neurons 1]);

Wxo_1=unifrnd(lowerb,upperb,[n1_neurons n0_neurons+1]);
Who_1=unifrnd(lowerb,upperb,[n1_neurons 1]);

Wxf_1=unifrnd(lowerb,upperb,[n1_neurons n0_neurons+1]);
Whf_1=unifrnd(lowerb,upperb,[n1_neurons 1]);

Wxc_1=unifrnd(lowerb,upperb,[n1_neurons n0_neurons+1]);
Whc_1=unifrnd(lowerb,upperb,[n1_neurons 1]);

neti_1=zeros(n1_neurons,1);
neto_1=zeros(n1_neurons,1);
netf_1=zeros(n1_neurons,1);
netc_1=zeros(n1_neurons,1);
h1=zeros(n1_neurons,1);
f1=zeros(n1_neurons,1);
i1=zeros(n1_neurons,1);
o1=zeros(n1_neurons,1);
c1_tilda=0.5*ones(n1_neurons,1);
c1=0.5*ones(n1_neurons,1);

%% second LSTM
Wxi_2=unifrnd(lowerb,upperb,[n2_neurons n1_neurons+1]);
Whi_2=unifrnd(lowerb,upperb,[n2_neurons 1]);

Wxo_2=unifrnd(lowerb,upperb,[n2_neurons n1_neurons+1]);
Who_2=unifrnd(lowerb,upperb,[n2_neurons 1]);

Wxf_2=unifrnd(lowerb,upperb,[n2_neurons n1_neurons+1]);
Whf_2=unifrnd(lowerb,upperb,[n2_neurons 1]);

Wxc_2=unifrnd(lowerb,upperb,[n2_neurons n1_neurons+1]);
Whc_2=unifrnd(lowerb,upperb,[n2_neurons 1]);
neti_2=zeros(n2_neurons,1);
neto_2=zeros(n2_neurons,1);
netf_2=zeros(n2_neurons,1);
netc_2=zeros(n2_neurons,1);
h2=zeros(n2_neurons,1);
f2=zeros(n2_neurons,1);
i2=zeros(n2_neurons,1);
o2=zeros(n2_neurons,1);
c2_tilda=0.5*ones(n2_neurons,1);
c2=0.5*ones(n2_neurons,1);
%% initial memory
%     h1_z1=zeros(n1_neurons,1);
%     f1_z1=zeros(n1_neurons,1);
%     i1_z1=zeros(n1_neurons,1);
%     c1_tilda_z1=0.5*ones(n1_neurons,1);
%     c1_z1=0.5*ones(n1_neurons,1);
%     o1_z1=zeros(n1_neurons,1);
%     f_derivative_i1_z1=zeros(n1_neurons,n1_neurons);
%     f_derivative_c1_z1=zeros(n1_neurons,n1_neurons);
%     X1_z1=zeros((n1_neurons+n0_neurons+1),1);
%     
%     h2_z1=zeros(n2_neurons,1);
%     f2_z1=zeros(n2_neurons,1);
%     i2_z1=zeros(n2_neurons,1);
%     o2_z1=zeros(n2_neurons,1);
%     c2_tilda_z1=0.5*ones(n2_neurons,1);
%     c2_z1=0.5*ones(n2_neurons,1);
%     f_derivative_i2_z1=zeros(n2_neurons,n2_neurons);
%     f_derivative_c2_z1=zeros(n2_neurons,n2_neurons);
%     X2_z1=zeros((n2_neurons+n1_neurons+1),1);
%% Output layer/Mlp
train_rate=0.7;
[n m]=size(DATA);
num_of_train=round(train_rate*n);
num_of_test=n-num_of_train;
data_train=DATA(1:num_of_train,:);
data_test=DATA(num_of_train+1:n,:);
target_train=TARGET(1:num_of_train,:);
target_test=TARGET(num_of_train+1:n,:);

W_3=unifrnd(lowerb,upperb,[l1_neurons n2_neurons]);
net3=zeros(l1_neurons,1);
o3=zeros(l1_neurons,1);

error_train=zeros(num_of_train,3);
error_test=zeros(num_of_test,3);

output_train=zeros(num_of_train,3);
output_test=zeros(num_of_test,3);

mse_train1=zeros(epochs_p,1);
mse_train2=zeros(epochs_p,1);
mse_train3=zeros(epochs_p,1);

mse_test1=zeros(epochs_p,1);
mse_test2=zeros(epochs_p,1);
mse_test3=zeros(epochs_p,1);

%%
for i=1:epochs_p
    %% initial memory

    h1_z1=zeros(n1_neurons,1);
    f1_z1=zeros(n1_neurons,1);
    i1_z1=zeros(n1_neurons,1);
    c1_tilda_z1=0.5*ones(n1_neurons,1);
    c1_z1=0.5*ones(n1_neurons,1);
    o1_z1=zeros(n1_neurons,1);
    f_derivative_i1_z1=zeros(n1_neurons,n1_neurons);
    f_derivative_c1_z1=zeros(n1_neurons,n1_neurons);
    X1_z1=zeros((n1_neurons+n0_neurons+1),1);
    
    h2_z1=zeros(n2_neurons,1);
    f2_z1=zeros(n2_neurons,1);
    i2_z1=zeros(n2_neurons,1);
    o2_z1=zeros(n2_neurons,1);
    c2_tilda_z1=0.5*ones(n2_neurons,1);
    c2_z1=0.5*ones(n2_neurons,1);
    f_derivative_i2_z1=zeros(n2_neurons,n2_neurons);
    f_derivative_c2_z1=zeros(n2_neurons,n2_neurons);
    X2_z1=zeros((n2_neurons+n1_neurons+1),1);
    
    for j=1:num_of_train
         input=data_train(j,1:m);
         target=target_train(j,1:3);
    %% first LSTM layer
       % network input 
        X1=[h1_z1;input';1];
        
        % input gate
        Wi_1=[diag(Whi_1) Wxi_1];
        neti_1=Wi_1*X1;
        i1=logsig(neti_1);
        
        % forget gate
        Wf_1=[diag(Whf_1) Wxf_1];
        netf_1=Wf_1*X1;
        f1=logsig(netf_1);
        %long term memory
        Wc_1=[diag(Whc_1) Wxc_1];
        netc_1=Wc_1*X1;
        c1_tilda=tansig(netc_1);
        c1=diag(f1)*c1_z1+diag(i1)*c1_tilda;
        
        % output gate
        Wo_1=[diag(Who_1) Wxo_1];
        neto_1=Wo_1*X1;
        o1=logsig(neto_1);
        
        % short term memory
        h1=diag(o1)*tansig(c1);
       %% Second LSTM layer
        % network input 
        X2=[h2_z1;h1;1];

        % input gate
        Wi_2=[diag(Whi_2) Wxi_2];
        neti_2=Wi_2*X2;
        i2=logsig(neti_2);
       
        % forget gate
        Wf_2=[diag(Whf_2) Wxf_2];
        netf_2=Wf_2*X2;
        f2=logsig(netf_2);
        
        % long term memory
        Wc_2=[diag(Whc_2) Wxc_2];
        netc_2=Wc_2*X2;
        c2_tilda=tansig(netc_2);
        c2=diag(f2)*c2_z1+diag(i2)*c2_tilda;
        
        % output gate
        Wo_2=[diag(Who_2) Wxo_2];

        neto_2=Wo_2*X2;
        o2=logsig(neto_2);
        
        % short term memory
        h2=diag(o2)*tansig(c2);
        %% output layer/mlp
         net3=W_3*h2;
         o3=logsig(net3);
         output_train(j,:)=o3';
        % Calculate error
          e=target-o3';
          error_train(j,:)=e;
          
        % Back Propagation
        f_derivative_i1=diag(i1.*(1-i1));
        f_derivative_o1=o1.*(1-o1);
        f_derivative_f1=diag(f1.*(1-f1));
        f_derivative_c1=diag((1-c1.^2));
        
        f_derivative_i2=diag(i2.*(1-i2));
        f_derivative_o2=o2.*(1-o2);
        f_derivative_f2=diag(f2.*(1-f2));
        f_derivative_c2=diag((1-c2.^2));
        A=diag(o3.*(1-o3));
        
        %third layer
          delta_W_3= eta_p*(e*A)'*h2';
        %second layer
          delta_Wi_2=(eta_p  * e * A * W_3)'*(o2.*(1-tansig(c2).^2))'*(f2.*f_derivative_i2_z1*c2_tilda_z1*X2_z1'+f_derivative_i2*c2_tilda*X2');
          delta_Wo_2=(eta_p  * e * A * W_3)*tansig(c2)* f_derivative_o2*X2';
          delta_Wf_2=(eta_p  * e * A * W_3)'*(o2.*(1-tansig(c2).^2))'*(c2_z1'*f_derivative_f2)'*X2';
          delta_Wc_2=(eta_p  * e * A * W_3)'*(o2.*(1-tansig(c2).^2))'*(f2.*f_derivative_c2_z1*i2_z1*X2_z1'+f_derivative_c2*i2*X2');
         %first layer
           Wh_1=Wo_2(1:n2_neurons,n2_neurons+1:n2_neurons+n1_neurons);
           delta_Wi_1=(eta_p  * e * A * W_3*(tansig(c2).*f_derivative_o2')*Wh_1)'*(o1.*(1-tansig(c1).^2))'*(f1.*f_derivative_i1_z1*c1_tilda_z1*X1_z1'+f_derivative_i1*c1_tilda*X1');
           delta_Wo_1=(eta_p  * e * A * W_3*(tansig(c2).*f_derivative_o2')*Wh_1)'*(tansig(c1)'*f_derivative_o1)*X1';
           delta_Wf_1=(eta_p  * e * A * W_3*(tansig(c2).*f_derivative_o2')*Wh_1)'*(o1.*(1-tansig(c1).^2))'*(c1_z1'*f_derivative_f1)'*X1';
           delta_Wc_1=(eta_p  * e * A * W_3*(tansig(c2).*f_derivative_o2')*Wh_1)'*(o1.*(1-tansig(c1).^2))'*(f1.*f_derivative_c1_z1*i1_z1*X1_z1'+f_derivative_c1*i1*X1');
%        
        % apdate weights

         W_3 = W_3 + delta_W_3;          
         Wi_2 = Wi_2 + delta_Wi_2;
         Wo_2 = Wo_2 + delta_Wo_2;
         Wf_2 = Wf_2 + delta_Wf_2;
         Wc_2 = Wc_2 + delta_Wc_2;
       
        Wi_1 = Wi_1 + delta_Wi_1;
        Wo_1 = Wo_1 + delta_Wo_1;
        Wf_1 = Wf_1 + delta_Wf_1;
        Wc_1 = Wc_1 + delta_Wc_1;

          % first Lstm Memory
          h1_z1=h1;
          c1_z1=c1;
          i1_z1=i1;
          o1_z1=o1;
          f1_z1=f1;
          X1_z1=X1;
          c1_tilda_z1=c1_tilda;
          f_derivative_i1_z1=f_derivative_i1;
          f_derivative_c1_z1=f_derivative_c1;
          % Second Lstm Memory
          h2_z1=h2;
          c2_z1=c2;
          i2_z1=i2;
          o2_z1=o2;
          f2_z1=f2;
          X2_z1=X2;
          c2_tilda_z1=c2_tilda;
          f_derivative_i2_z1=f_derivative_i2;
          f_derivative_c2_z1=f_derivative_c2;
    end
%       Mean square train error
      mse_train1(i,1)=mse(error_train(:,1));
      mse_train2(i,1)=mse(error_train(:,2));
      mse_train3(i,1)=mse(error_train(:,3));

% 
    h1_z1=zeros(n1_neurons,1);
    c1_z1=0.5*ones(n1_neurons,1);
    h2_z1=zeros(n2_neurons,1);
    c2_z1=0.5*ones(n2_neurons,1);
%     
for j=1:num_of_test
         input=data_test(j,1:m);
         target=target_test(j,1:3);
%     % network input 
        X1=[h1_z1;input';1];
        
        % input gate
        Wi_1=[diag(Whi_1) Wxi_1];
        neti_1=Wi_1*X1;
        i1=logsig(neti_1);
        
        % forget gate
        Wf_1=[diag(Whf_1) Wxf_1];
        netf_1=Wf_1*X1;
        f1=logsig(netf_1);
        %long term memory
        Wc_1=[diag(Whc_1) Wxc_1];
        netc_1=Wc_1*X1;
        c1_tilda=tansig(netc_1);
        c1=diag(f1)*c1_z1+diag(i1)*c1_tilda;
        
        % output gate
        Wo_1=[diag(Who_1) Wxo_1];
        neto_1=Wo_1*X1;
        o1=logsig(neto_1);
        
        % short term memory
        h1=diag(o1)*tansig(c1);
       %% Second LSTM layer
        X2=[h2_z1;h1;1];

        % input gate
        Wi_2=[diag(Whi_2) Wxi_2];
        neti_2=Wi_2*X2;
        i2=logsig(neti_2);
       
        % forget gate
        Wf_2=[diag(Whf_2) Wxf_2];
        netf_2=Wf_2*X2;
        f2=logsig(netf_2);
        
        % long term memory
        Wc_2=[diag(Whc_2) Wxc_2];
        netc_2=Wc_2*X2;
        c2_tilda=tansig(netc_2);
        c2=diag(f2)*c2_z1+diag(i2)*c2_tilda;
        
        % output gate
        Wo_2=[diag(Who_2) Wxo_2];
        neto_2=Wo_2*X2;
        o2=logsig(neto_2);
        
        % short term memory
        h2=diag(o2)*tansig(c2);
        %% output layer/mlp
         net3=W_3*h2;
         o3=logsig(net3);
         output_test(j,:)=o3';
        % Calculate error
          e=target-o3';
          error_test(j,:)=e;

    h1_z1=h1;
    c1_z1=c1;
    h2_z1=h2;
    c2_z1=c2;

end   
%   % Mean square test error
      mse_test1(i,1)=mse(error_test(:,1));
      mse_test2(i,1)=mse(error_test(:,2));
      mse_test3(i,1)=mse(error_test(:,3));      
  %% Plot Results
%        Find Regression
  [m_train1 ,b_train1]=polyfit(target_train(:,1),output_train(:,1),1);
  [y_fit_train1,~] = polyval(m_train1,target_train(:,1),b_train1);
  [m_test1 ,b_test1]=polyfit(target_test(:,1),output_test(:,1),1);
  [y_fit_test1,~] = polyval(m_test1,target_test(:,1),b_test1);
  figure(1);
  subplot(2,3,1),plot(target_train(:,1),'-r');
  hold on;
  subplot(2,3,1),plot(output_train(:,1),'-b');
  title('Output Train')
  hold off;
  
  subplot(2,3,2),semilogy(mse_train1(1:i,1),'-r');
  title('MSE Train')
  hold off;
  
  subplot(2,3,3),plot(target_train(:,1),output_train(:,1),'b*');hold on;
  plot(target_train(:,1),y_fit_train1,'r-');
  title('Regression Train')
  hold off;
  
  subplot(2,3,4),plot(target_test(:,1),'-r');
  hold on;
  subplot(2,3,4),plot(output_test(:,1),'-b');
  title('Output Test')
  hold off;
  
  subplot(2,3,5),plot(mse_test1(1:i,1),'-r');
  title('MSE Test')
  hold off;
  
  subplot(2,3,6),plot(target_test(:,1),output_test(:,1),'b*');hold on;
  plot(target_test(:,1),y_fit_test1,'r-');
  title('Regression Test')
  hold off;
   %% Plot Results
      % Find Regression
  [m_train2 ,b_train2]=polyfit(target_train(:,2),output_train(:,2),1);
  [y_fit_train2,~] = polyval(m_train2,target_train(:,2),b_train2);
  [m_test2 ,b_test2]=polyfit(target_test(:,2),output_test(:,2),1);
  [y_fit_test2,~] = polyval(m_test2,target_test(:,2),b_test2);
  figure(2);
  subplot(2,3,1),plot(target_train(:,2),'-r');
  hold on;
  subplot(2,3,1),plot(output_train(:,2),'-b');
  title('Output Train')
  hold off;
%   
  subplot(2,3,2),semilogy(mse_train2(1:i,1),'-r');
  title('MSE Train')
  hold off;
  
  subplot(2,3,3),plot(target_train(:,2),output_train(:,2),'b*');hold on;
  plot(target_train(:,2),y_fit_train2,'r-');
  title('Regression Train')
  hold off;
  
  subplot(2,3,4),plot(target_test(:,2),'-r');
  hold on;
  subplot(2,3,4),plot(output_test(:,2),'-b');
  title('Output Test')
  hold off;
  
  subplot(2,3,5),plot(mse_test2(1:i,1),'-r');
  title('MSE Test')
  hold off;
  
  subplot(2,3,6),plot(target_test(:,2),output_test(:,2),'b*');hold on;
  plot(target_test(:,2),y_fit_test2,'r-');
  title('Regression Test')
  hold off;
%  %% Plot Results
%        % Find Regression

  [m_train3 ,b_train3]=polyfit(target_train(:,3),output_train(:,3),1);
  [y_fit_train3,~] = polyval(m_train3,target_train(:,3),b_train3);
  [m_test3 ,b_test3]=polyfit(target_test(:,3),output_test(:,3),1);
  [y_fit_test3,~] = polyval(m_test3,target_test(:,3),b_test3);
  figure(3);
  subplot(2,3,1),plot(target_train(:,3),'-r');
  hold on;
  subplot(2,3,1),plot(output_train(:,3),'-b');
  title('Output Train')
  hold off;
%   
  subplot(2,3,2),semilogy(mse_train3(1:i,1),'-r');
  title('MSE Train')
  hold off;
  
  subplot(2,3,3),plot(target_train(:,3),output_train(:,3),'b*');hold on;
  plot(target_train(:,3),y_fit_train3,'r-');
  title('Regression Train')
  hold off;
  
  subplot(2,3,4),plot(target_test(:,3),'-r');
  hold on;
  subplot(2,3,4),plot(output_test(:,3),'-b');
  title('Output Test')
  hold off;
  
  subplot(2,3,5),plot(mse_test3(1:i,1),'-r');
  title('MSE Test')
  hold off;
  
  subplot(2,3,6),plot(target_test(:,3),output_test(:,3),'b*');hold on;
  plot(target_test(:,3),y_fit_test3,'r-');
  title('Regression Test')
  hold off;
  pause(0.001);
end

 fprintf('mse train 1 = %1.16g, mse test 1 = %1.16g \n', mse_train1(epochs_p,1), mse_test1(epochs_p,1))
 fprintf('mse train 2 = %1.16g, mse test 2 = %1.16g \n', mse_train2(epochs_p,1), mse_test2(epochs_p,1))
 fprintf('mse train 3 = %1.16g, mse test 3 = %1.16g \n', mse_train3(epochs_p,1), mse_test3(epochs_p,1))

