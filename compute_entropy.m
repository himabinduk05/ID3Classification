function entropy = compute_entropy(data, nbins)
%compute_entropy returns the entropy of a sequence
%                data is a column vector of values
%                nbins is used for discretization
%	 GourdelKanakamedalaMa

    p = hist(data(:), nbins);
    p = p / sum(p(:));
    
    % i = find(p); % remove 0 values
    %entropy = -sum(p(i) .* log2(p(i))); % Compute entropy
    entropy = -log2(p)*p';
end