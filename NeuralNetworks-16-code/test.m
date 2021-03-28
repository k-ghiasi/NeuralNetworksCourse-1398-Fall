% Boltzmann algorithm for encoder problem
clc;
clear;

% s = RandStream('mcg16807','Seed',0);
% RandStream.setGlobalStream(s);

v = [1 0 0 0 1 0 0 0; 0 1 0 0 0 1 0 0; 0 0 1 0 0 0 1 0; 0 0 0 1 0 0 0 1];
%v(v == 0) = -1;
%vv = [repmat(v(1,:),25,1); repmat(v(2,:),25,1); repmat(v(3,:),25,1); repmat(v(4,:),25,1)];
%v = vv;
[adj, places] = CreateDecoderAdjacencyMat(4, 2);
schdule = [2 20; 2 15; 2 12; 4 10];
%schdule = [ 40 20; 40 15; 40 12; 40 10];
%schdule = [ 40 20; 40 15; 40 12; 40 10;40 8;40 6; 40 5];
pNoise = [0.05 0.15];
%pNoise = [0.005 0.015];
%pNoise = [0 0];
%pNoise = [0.005 0.015];

b1 = BoltzmannMachine(8, 2, adj, 'binary');
%b1 = b1.TrainBatch(v, 2000, 2, schdule, 10, 'static', 2, pNoise);
b1 = b1.TrainBatch(v, 200, 100, schdule, 1, 'static', 2, pNoise);
img = b1.Draw(places, 1);
imshow(img);

schduleGentle = [ 40 20; 40 15; 40 12; 40 10;40 8;40 6; 40 5];
v = b1.Complete([1 0 0 0 0 0 0 0], [1 1 1 1 0 0 0 0 0 0], schduleGentle);
disp(v);

v = b1.Complete([0 1 0 0 0 0 0 0], [1 1 1 1 0 0 0 0 0 0], schduleGentle);
disp(v);

v = b1.Complete([0 0 1 0 0 0 0 0], [1 1 1 1 0 0 0 0 0 0], schduleGentle);
disp(v);

v = b1.Complete([0 0 0 1 0 0 0 0], [1 1 1 1 0 0 0 0 0 0], schduleGentle);
disp(v);
