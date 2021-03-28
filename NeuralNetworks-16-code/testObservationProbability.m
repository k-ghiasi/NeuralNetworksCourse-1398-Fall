load('b1_25_4.mat');
v = [1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1];
d = zeros(1, 4);
%schduleGentle = [ 40 20; 40 15; 40 12; 40 10;40 8;40 6; 40 5];
schduleGentle = ones (2,30)*100;
schduleGentle(2,1:30)=30:-1:1;
for i=1:1000
    vv = b1.Complete([0 0 0 0 0 0 0 0], [0 0 0 0 0 0 0 0], schduleGentle);
    for j=1:4
        if(sum(v(j, :) == vv(1:4)) == 4)
            d(j) = d(j)+1;
            break;
        end
    end
end
dd = (d / sum(d)) * 100;
disp(dd);