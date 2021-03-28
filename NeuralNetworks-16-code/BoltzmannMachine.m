% File: BoltzmannMachine.m
%
% Copyright (C) Kamaledin Ghiasi-Shirazi, Ferdowsi Univerity of Mashhad, 2014 (1393 Hijri Shamsi)
%
% Authors: 	Amir Ahooye Atashin
%           Kamaledin Ghiasi-Shirazi

classdef BoltzmannMachine
    %BoltzmannMachine
    
    properties(GetAccess = 'public', SetAccess = 'private')
        nv;
        nh;
        n;
        w;
        adjMat;
        inactiveState = -1; %default bipolar neuron
      
        weightSquareSize = 6;
        marginBetweenWeights = 2;
        XMarginBetweenNeurons = 10;
        YMarginBetweenNeurons = 30;
    end
    
    properties(GetAccess = 'public', SetAccess = 'public')
        showGoingToEquilibriumState = 0;
    end
    
    properties(Constant = true)
    end
    
    methods
        %constructor
        function obj = BoltzmannMachine(nv, nh, adjMat, mode)
            obj.nv = nv;
            obj.nh = nh;
            obj.n = nv+nh;
            obj.adjMat = adjMat;
            obj.adjMat = obj.adjMat + diag(ones(1,obj.n));
            obj.w = zeros(obj.n, obj.n);
            if(strcmp(mode, 'binary'))
                obj.inactiveState = 0;
            elseif(strcmp(mode, 'bipolar'))
                obj.inactiveState = -1;
            else
                error ('incorrect mode: %s', mode);
            end
        end
        
        %% Train Boltzmann Machine
        function obj = TrainBatch(obj, data, epochs, nEnsemble, annealingSchedule, statisticsCollectionTimes, learningMethod, learningRate, noiseProb)
            %% init
            nData = length(data(:,1));
            %% Main cycle
            for iep=1:epochs
                %for each data the network goes to equilibrium state
                pSum = zeros(obj.n, obj.n);
                clampedUnits = zeros(1, obj.n);
                clampedUnits(1:obj.nv) = ones(1, obj.nv);                    
                for i=1:nData
                    vNoisy = obj.AddNoise(data(i, :), noiseProb);
                    hClamped = zeros(nEnsemble, obj.nh);
                    pClamped = zeros(nEnsemble, obj.n, obj.n);
                    for j=1:nEnsemble
                        [~, hClamped(j,:)] = obj.GoToEquilibriumState(vNoisy, clampedUnits, annealingSchedule);
                        pClamped(j,:,:) = obj.CollectStatics(vNoisy, hClamped(j,:), clampedUnits, statisticsCollectionTimes, annealingSchedule(end,2));
                        pSum = pSum + squeeze(pClamped(j,:,:));
                    end
                end
                pClampedAvg = pSum / (nData * nEnsemble);
                clampedUnits = zeros(1, obj.n);
                vDummy = zeros(1, obj.nv); % since the values will be overwritten in GoToEquilibriumState
                
                pSumFree = zeros(obj.n, obj.n);
                vFree = zeros(nEnsemble, obj.nv);
                hFree = zeros(nEnsemble, obj.nh);
                pFree = zeros(nEnsemble, obj.n, obj.n);
                for j=1:nEnsemble
                    [vFree(j,:), hFree(j,:)] = obj.GoToEquilibriumState(vDummy, clampedUnits, annealingSchedule);
                    pFree(j,:,:) = obj.CollectStatics(vFree(j,:), hFree(j,:), clampedUnits, statisticsCollectionTimes, annealingSchedule(end,2));
                    pSumFree = pSumFree + squeeze(pFree(j,:,:));
                end
                pFreeAvg = pSumFree / nEnsemble;
                
                %sum(sum(squeeze(std(pFree,1))))
                %pClampedAvg-pFreeAvg
                %%update weights
                s = pClampedAvg-pFreeAvg;
                if(strcmp(learningMethod, 'static'))
                    s = sign(s);
                end
                s = s .* obj.adjMat;
                obj.w = obj.w + learningRate * s;

                obj.w
            end
        end
        
        function v = Complete(obj, v, clampedUnits, annealingSchedule)
            [v, ~] = GoToEquilibriumState(obj, v, clampedUnits, annealingSchedule);
        end
        %% Draw
        function [img] = Draw(obj, placeMap, shape)
            neuralSquareSize = obj.ComputeNeuralSquareSize(placeMap);
            
            imgSize = neuralSquareSize .* size(placeMap);
            img = ones(imgSize) * 128;
            for i=1:obj.n
                [j, k] = find(placeMap == i);
                if(shape == 1)
                    neuralSquareImage = obj.DrawWeightsSquare(i, placeMap);
                else
                    neuralSquareImage = obj.DrawWeightsSquare2(i, placeMap);
                end
                width = size(neuralSquareImage,1);
                height = size(neuralSquareImage,2);
                x = (j-1) * width;
                y = (k-1) * height;
                img(x+1:x+width, y+1:y+height) = neuralSquareImage;
            end
            img = mat2gray(img);
        end
        
        function neuralSquareSize = ComputeNeuralSquareSize(obj, placeMap)
            neuralSquareSize = (obj.weightSquareSize+obj.marginBetweenWeights) * size(placeMap) + [obj.XMarginBetweenNeurons obj.YMarginBetweenNeurons];
        end
        
        function [img] = DrawWeightsSquare(obj, ind, placeMap)
            maxw = max(max(abs(obj.w)));
            neuralSquareSize = obj.ComputeNeuralSquareSize(placeMap);
            img = ones(neuralSquareSize) * 127;
            
            for i=1:obj.n
                if obj.adjMat(ind, i) == 1
                    
                    weight = obj.w(ind, i);
                    sz = obj.weightSquareSize;
                    color = 127 + weight/maxw * 127;
                    imgw = ones(sz,sz) * color;
                    
                    [x,y] = find (placeMap==i);
                    x = (x-1) * (obj.weightSquareSize+obj.marginBetweenWeights) + obj.marginBetweenWeights/2;
                    y = (y-1) * (obj.weightSquareSize+obj.marginBetweenWeights) + obj.marginBetweenWeights/2;
                    
                    x = x + obj.XMarginBetweenNeurons / 2;
                    y = y + obj.YMarginBetweenNeurons / 2;
                    img(x:x+sz-1,y:y+sz-1) = imgw;
                end
            end
        end
        
        function [img] = DrawWeightsSquare2(obj, ind, placeMap)
            maxw = max(max(abs(obj.w)));
            neuralSquareSize = obj.ComputeNeuralSquareSize(placeMap);
            img = ones(neuralSquareSize) * 127;
            
            for i=1:obj.n
                if obj.adjMat(ind, i) == 1
                    color = 1;
                    weight = obj.w(ind, i);
                    if(weight < 0)
                        weight = -weight;
                        color = 255;
                    end
                    sz = round(obj.weightSquareSize * weight/maxw);
                    imgw = ones(obj.weightSquareSize, obj.weightSquareSize) * 127;
                    d = obj.weightSquareSize - sz;
                    imgw(1+d:end-d,1+d:end-d) = color;
                    
                    [x,y] = find (placeMap==i);
                    x = (x-1) * (obj.weightSquareSize+obj.marginBetweenWeights) + obj.marginBetweenWeights/2;
                    y = (y-1) * (obj.weightSquareSize+obj.marginBetweenWeights) + obj.marginBetweenWeights/2;
                    
                    x = x + obj.XMarginBetweenNeurons / 2;
                    y = y + obj.YMarginBetweenNeurons / 2;
                    sz = obj.weightSquareSize;
                    img(x:x+sz-1,y:y+sz-1) = imgw;
                end
            end
        end
    end
    methods(Access = private)
        %% Add noise for noisy clamping technic
        function v = AddNoise(obj, v, prob)
            for i=1:obj.nv
                if(v(i) == 1)
                    if(rand() <= prob(2))
                        v(i) = obj.inactiveState;
                    end
                else
                    if(rand() <= prob(1))
                        v(i) = 1;
                    end
                end
            end
        end
        
        %%
        function [v, h] = GoToEquilibriumState(obj, v, clamped, annealingSchedule)
            vhInit = [v, zeros(1,obj.nh)];
            vhRandom = obj.GenerateRandomVector(obj.n);
            vh = clamped .* vhInit + (1-clamped) .* vhRandom;
            
            for t=1:length(annealingSchedule(:, 1))
                schedule = annealingSchedule(t, :);
                T = schedule(2);
                for s=1:schedule(1)
                    perm = randperm(obj.n);
                    maxP = 0;
                    for jj=1:obj.n
                        j = perm(jj);
                        if(clamped(j) == 0) %do if unclamped
                            vh_j_old = vh(j);
                            vh(j) = 1; % to act as bias
                            sum = obj.w(j, :) * vh';
                            p = 1 / (1 + exp(-sum/T));
                            if (vh_j_old == 0 && p > maxP)
                                maxP = p;
                            end
                            if (vh_j_old == 1 && (1-p) > maxP)
                                maxP = 1-p;
                            end
                            if rand() <= p
                                vh(j) = 1;
                            else
                                vh(j) = obj.inactiveState;
                            end
                        end
                    end
                    if (obj.showGoingToEquilibriumState)
                         display (maxP);
                         txt = sprintf('temprature=%d', T);
                         display (txt);
                         vh
                    end
                end
            end
            
            v = vh(1:obj.nv);
            h = vh(obj.nv+1:end);
        end
        %%
        function [stats] = CollectStatics(obj, v, h, clamped, timeUnits, T)
            count = 0;
            vh = [v, h];
            stats = zeros(obj.n, obj.n);
            perm = randperm(obj.n);            
            for t=1:timeUnits
                for jj=1:obj.n
                    j = perm(jj);
                    if(j > obj.nv || clamped(j) == 0) %do if unclamped
                        vh(j) = 1; % to act as bias
                        sum = obj.w(j, :) * vh';
                        p = 1 / (1 + exp(-sum/T));
                        if rand() <= p
                            vh(j) = 1;
                        else
                            vh(j) = obj.inactiveState;
                        end    
                    end
                end
                vhBinary = vh;
                vhBinary(vhBinary == -1) = 0;
                stats = stats + vhBinary' * vhBinary;
                count = count + 1;
            end
            stats = stats / count;
        end
        
        function [vRandFinal] = GenerateRandomVector(obj, n)
            vRand = randi(2, n, 1)';
            vRand1 =  vRand - 1;
            vRand1(vRand1 == 0) = obj.inactiveState;
            
            vRand2 = vRand;
            vRand2(vRand2 == 2) = obj.inactiveState;
            vRandFinal = vRand1;
        end
    end
end
