% Main file
% Machine learning

% KANAKAMEDALA Hima Bindu


close all; clear all; clc;

%% Initialization

load iris.dat % data
nb_attributes = 4;
nb_classes = 3;
nbins = 3; 

% slection of node
node = select_node(iris, nbins, nb_classes, nb_attributes)



