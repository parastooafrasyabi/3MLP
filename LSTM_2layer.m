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

%%  Making Data
 x=X;
 N = size(x(:,1),1);
    D=5;
  L=0;;
[Data1,Target1]=Make_Data(x(:,1),N,D,L);
 N = size(x(:,2),1);
    D=5;
  L=0;
[Data2,Target2]=Make_Data(x(:,2),N,D,L);
 N = size(x(:,3),1);
    D=5;
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
n1_neurons=64;
l1_neurons=3;
lowerb=-1;
upperb=1;
epochs_p=1000;
eta_p=0.01;
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
%% initial memory
% h1_z1=zeros(n1_neurons,1);
% f1_z1=zeros(n1_neurons,1);
% i1_z1=zeros(n1_neurons,1);
% c1_tilda_z1=0.5*ones(n1_neurons,1);
% c1_z1=0.5*ones(n1_neurons,1);
% o1_z1=zeros(n1_neurons,1);
% f_derivative_i1_z1=zeros(n1_neurons,n1_neurons);
% f_derivative_c1_z1=zeros(n1_neurons,n1_neurons);
% X1_z1=zeros((n1_neurons+n0_neurons+1),1);

 
   
%% Output layer/Mlp
train_rate=0.7;
[n m]=size(DATA);
num_of_train=round(train_rate*n);
num_of_test=n-num_of_train;
data_train=DATA(1:num_of_train,:);
data_test=DATA(num_of_train+1:n,:);
target_train=TARGET(1:num_of_train,:);
target_test=TARGET(num_of_train+1:n,:);

W_2=unifrnd(lowerb,upperb,[l1_neurons n1_neurons]);
net2=zeros(l1_neurons,1);
o2=zeros(l1_neurons,1);
% g2=unifrnd(0,1,[l1_neurons,1]);

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
        %% output layer/mlp
         net2=W_2*h1;
         o2=logsig(net2);
%          o2=logsig(g2.*net2);
         output_train(j,:)=o2';
        % Calculate error
          e=target-o2';
          error_train(j,:)=e;
          
        % Back Propagation
        f_derivative_i1=diag(i1.*(1-i1));
        f_derivative_o1=o1.*(1-o1);
        f_derivative_f1=diag(f1.*(1-f1));
        f_derivative_c1=diag((1-c1.^2));
        A=diag(o2.*(1-o2));
%         A=diag(g2.*(o2.*(1-o2)));
%         df_g2=diag(net2.*(o2.*(1-o2)));

          delta_Wi_1=(eta_p  * e * A * W_2)'*(o1.*(1-tansig(c1).^2))'*(f1.*f_derivative_i1_z1*c1_tilda_z1*X1_z1'+f_derivative_i1*c1_tilda*X1');
          delta_Wo_1=(eta_p  * e * A * W_2)*tansig(c1)* f_derivative_o1*X1';
          delta_Wf_1=(eta_p  * e * A * W_2)'*(o1.*(1-tansig(c1).^2))'*(c1_z1'*f_derivative_f1)'*X1';
          delta_Wc_1=(eta_p  * e * A * W_2)'*(o1.*(1-tansig(c1).^2))'*(f1.*f_derivative_c1_z1*i1_z1*X1_z1'+f_derivative_c1*i1*X1');
          delta_W_2= eta_p*(e*A)'*h1';
          %
%           delta_Wi_1=(eta_p  * e * A * W_2)'*(o1.*(1-tansig(c1).^2))'*(f1.*f_derivative_i1_z1*c1_tilda_z1*X1_z1'+f_derivative_i1*c1_tilda*X1');
%           delta_Wo_1=(eta_p  * e * A * W_2)*tansig(c1)* f_derivative_o1*X1';
%           delta_Wf_1=(eta_p  * e * A * W_2)'*(o1.*(1-tansig(c1).^2))'*(c1_z1'*f_derivative_f1)'*X1';
%           delta_Wc_1=(eta_p  * e * A * W_2)'*(o1.*(1-tansig(c1).^2))'*(f1.*f_derivative_c1_z1*i1_z1*X1_z1'+f_derivative_c1*i1*X1');
%           delta_W_2= eta_p*(e*A)'*h1';
%           delta_g2=eta_p*(e*df_g2)';

        % apdate weights

          Wi_1 = Wi_1 + delta_Wi_1;
          Wo_1 = Wo_1 + delta_Wo_1;
          Wf_1 = Wf_1 + delta_Wf_1;
          Wc_1 = Wc_1 + delta_Wc_1;
          W_2 = W_2 + delta_W_2;             
%           g2 = g2 + delta_g2;             

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
        
    end
%       Mean square train error
      mse_train1(i,1)=mse(error_train(:,1));
      mse_train2(i,1)=mse(error_train(:,2));
      mse_train3(i,1)=mse(error_train(:,3));

% 
    h1_z1=zeros(n1_neurons,1);
    c1_z1=0.5*ones(n1_neurons,1);
 
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
        %% output layer/mlp
         net2=W_2*h1;
         o2=logsig(net2);
         output_test(j,:)=o2';
        % Calculate error
          e=target-o2';
          error_test(j,:)=e;

    h1_z1=h1;
    c1_z1=c1;
    
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
%    %% Plot Results
%       % Find Regression
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

