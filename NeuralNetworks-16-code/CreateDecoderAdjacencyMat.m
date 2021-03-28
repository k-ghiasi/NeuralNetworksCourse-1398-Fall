% File: CreateDecoderAdjacencyMat.m
%
% Copyright (C) Kamaledin Ghiasi-Shirazi, Ferdowsi Univerity of Mashhad, 2014 (1393 Hijri Shamsi)
%
% Authors: 	Amir Ahooye Atashin

function [ adj, placeMap ] = CreateDecoderAdjacencyMat( n, m )
%This function create decoder problem adjacency and place map matrix for a 
    adj = zeros(n * 2 + m, n * 2 + m);
    placeMap = zeros(3, n);
    
    %for v1
    for i=1:n
        adj(i, :) = 1;
        adj(i, i) = 0;
        adj(i, n+1:n+n) = 0;
    end
    %for v2
    for i=n+1:n+n
        adj(i, :) = 1;
        adj(i, i) = 0;
        adj(i, 1:n) = 0;
    end
    %for hiddens
    for i=n+n+1:n+n+m
        adj(i, :) = 1;
        adj(i, (n+n+1):end) = 0;
    end
    placeMap(1, :) = (1:n);
    placeMap(3, :) = (n+1:n+n);
    
    placeMap(2, 1+(n-m)/2:m+(n-m)/2) = (n+n+1:n+n+m);
end

