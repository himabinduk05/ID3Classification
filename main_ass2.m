% Main file
% Machine learning 

% KANAKAMEDALA Hima Bindu
 

close all; clear all; clc;

%% Initialization

load iris.dat % data
nb_attributes = 4;
attributes = [1 2 3 4]
nb_classes = 2;
nbins = 5;

% Discretization Step
for i = 1:nb_attributes
    att = iris(:,i);
    min_att(i) = min(att);
    max_att(i) = max(att);
    val(i) = (max_att(i) - min_att(i)) / nbins ;
end
iris_disc = iris;  
    % for each attribute
    for j = 1:nb_attributes

      % for each value
      for n = 1:size(iris_disc, 1)
      iris_disc(n,j) = ceil((iris_disc(n,j)- min_att(j))/val(j));
          if iris_disc(n,j) == 0;
                iris_disc(n,j) =1;
          end
      end
    end
    
% make setosa vs combined versicolor and virginica
for j = 1:size(iris_disc, 1)
    if iris_disc(j,5) ~= 1
       iris_disc(j,5) = 2;
    end
end

% Randomly divide the data set into two subset
x = linspace(1,size(iris_disc,1),size(iris_disc,1)); seq = randperm(numel(x));
iris_train = iris_disc(seq(1:75),:);
iris_test = iris_disc(seq(75+1:end),:);

%% Initialization for Naïve Bayes classifier
naive_train=iris_train;
naive_test=iris_test;
petal_length_values_count=zeros(nbins,2);
petal_width_values_count=zeros(nbins,2);
sepal_length_values_count=zeros(nbins,2);
sepal_width_values_count=zeros(nbins,2);
petal_length_values_prob=zeros(nbins,2);
petal_width_values_prob=zeros(nbins,2);
sepal_length_values_prob=zeros(nbins,2);
sepal_width_values_prob=zeros(nbins,2);
petal_length_setosa_count=0;
petal_width_setosa_count=0;
sepal_length_setosa_count=0;
sepal_width_setosa_count=0;
petal_length_vv_count=0;
petal_width_vv_count=0;
sepal_length_vv_count=0;
sepal_width_vv_count=0;
successcount=0;
failurecount=0;



%% TRAINING
for y=1:size(naive_train, 1)
    for binvalue=1:nbins
        if naive_train(y,1) == binvalue && naive_train(y,5) == 1
            petal_length_values_count(binvalue,1)=petal_length_values_count(binvalue,1)+1;
            petal_length_setosa_count=petal_length_setosa_count+1;
        end
        if naive_train(y,1) == binvalue && naive_train(y,5) == 2
            petal_length_values_count(binvalue,2)=petal_length_values_count(binvalue,1)+1;
            petal_length_vv_count=petal_length_vv_count+1;
        end  
        if naive_train(y,2) == binvalue && naive_train(y,5) == 1
            petal_width_values_count(binvalue,1)=petal_width_values_count(binvalue,1)+1;
            petal_width_setosa_count=petal_width_setosa_count+1;
        end
        if naive_train(y,2) == binvalue && naive_train(y,5) == 2
            petal_width_values_count(binvalue,2)=petal_width_values_count(binvalue,1)+1;   
            petal_width_vv_count=petal_width_vv_count+1;
        end
        if naive_train(y,3) == binvalue && naive_train(y,5) == 1
            sepal_length_values_count(binvalue,1)=sepal_length_values_count(binvalue,1)+1;
            sepal_length_setosa_count=sepal_length_setosa_count+1;
        end
        if naive_train(y,3) == binvalue && naive_train(y,5) == 2
            sepal_length_values_count(binvalue,2)=sepal_length_values_count(binvalue,1)+1;
            sepal_length_vv_count=sepal_length_vv_count+1;
        end
        if naive_train(y,4) == binvalue && naive_train(y,5) == 1
            sepal_width_values_count(binvalue,1)=sepal_width_values_count(binvalue,1)+1;
            sepal_width_setosa_count=sepal_width_setosa_count+1;
        end
        if naive_train(y,4) == binvalue && naive_train(y,5) == 2
            sepal_width_values_count(binvalue,2)=sepal_width_values_count(binvalue,1)+1; 
            sepal_width_vv_count=sepal_width_vv_count+1;
        end
    end
end
for row=1:nbins
    for col=1:2
        if col==1 
          petal_length_values_prob(row,col)=petal_length_values_count(row,col)/petal_length_setosa_count;
          petal_width_values_prob(row,col)=petal_width_values_count(row,col)/petal_width_setosa_count;
          sepal_length_values_prob(row,col)=sepal_length_values_count(row,col)/sepal_length_setosa_count;
          sepal_width_values_prob(row,col)=sepal_width_values_count(row,col)/sepal_width_setosa_count;
        else
          petal_length_values_prob(row,col)=petal_length_values_count(row,col)/petal_length_vv_count;
          petal_width_values_prob(row,col)=petal_width_values_count(row,col)/petal_width_vv_count;
          sepal_length_values_prob(row,col)=sepal_length_values_count(row,col)/sepal_length_vv_count;
          sepal_width_values_prob(row,col)=sepal_width_values_count(row,col)/sepal_width_vv_count;       
        end
    end
end

%% TESTING
for v=1:size(naive_test, 1)
    yes_prob=petal_length_values_prob(naive_test(v,1),1)*petal_width_values_prob(naive_test(v,2),1)*sepal_length_values_prob(naive_test(v,3),1)*sepal_width_values_prob(naive_test(v,4),1);
    no_prob=petal_length_values_prob(naive_test(v,1),2)*petal_width_values_prob(naive_test(v,2),2)*sepal_length_values_prob(naive_test(v,3),2)*sepal_width_values_prob(naive_test(v,4),2);
    if yes_prob > no_prob
       if naive_test(v,5)==1
           successcount=successcount+1;
       else
           failurecount=failurecount+1;
       end
    else
        if naive_test(v,5)==2
           successcount=successcount+1;
       else
           failurecount=failurecount+1;
        end
    end
end    
  
  Totalcount=successcount+failurecount;
 accuracy=(successcount/Totalcount)*100;
 disp(accuracy);









