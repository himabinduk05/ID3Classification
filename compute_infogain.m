function infogain = compute_infogain( data_set, nbins, entropyS )
%compute_infogain returns the information gain of an attribute among a data
%                 set
%                data_set is a column vector of values
%                nbins is used for discretization
%                entropyS is the entropy of the class of the data
%	 GourdelKanakamedalaMa

    infogain = entropyS - compute_entropy(data_set, nbins);
    
end