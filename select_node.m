function node = select_node(data, nbins, nb_classes, nb_attributes)
%select_node returns the attribute which has the highest information gain
%                data is a column vector of values
%                nbins is used for discretization
%	 GourdelKanakamedalaMa
    
    % Initialization
    infogain = zeros(1,4);
    entropyS = compute_entropy(data(:,nb_attributes+1), nb_classes);
    
    for i=1:nb_attributes
        infogain(1,i) = compute_infogain(data(:,i), nbins, entropyS)
    end

    [maxi, node] = max(infogain);
end