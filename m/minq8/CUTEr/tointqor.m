%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% tointqor.m %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Matlab script defining the data structure data containing the function
% parameters, the default box bounds xl and xu and the default starting
% point x for the CUTEr test set function TOINTQOR
%
n = 50;
data.gam = 0;
data.c = zeros(50,1);
data.D = 2*[1.25 1.4 2.4 1.4 1.75 1.2 2.25 1.2 1 1.1 1.5 1.6 1.25 1.25 1.2 1.2 1.4 0.5 0.5 1.25 1.8 0.75 1.25 1.4 1.6 2 1 1.6 1.25 2.75 1.25 1.25 1.25 3 1.5 2 1.25 1.4 1.8 1.5 2.2 1.4 1.5 1.25 2 1.5 1.25 1.4 0.6 1.5 1 1.5 1 0.1 1.5 2 1 1.5 3 2 1 3 0.1 1.5 0.15 2 1 0.1 3 0.1 1.2 1 0.1 2 1.2 3 1.5 3 2 1 1.2 2 1]';
data.b = [zeros(1,50) 5 5 5 2.5 6 6 5 6 10 6 5 9 2 7 2.5 6 5 2 9 2 5 5 2.5 5 6 10 7 10 6 5 4 4 4]';
B = zeros(33,50);
B(1,1) = -1;
B(1,31) = 1;
B(2,1) = 1;
B(2,[2 3]) = -1;
B(3,2) = 1;
B(3,[4 5]) = -1;
B(4,4) = 1;
B(4,[6 7]) = -1;
B(5,6) = 1;
B(5,[8 9]) = -1;
B(6,8) = 1;
B(6,[10 11]) = -1;
B(7,10) = 1;
B(7,[12 13]) = -1;
B(8,12) = 1;
B(8,[14 15]) = -1;
B(9,[11 13 14]) = 1;
B(9,[16 17]) = -1;
B(10,16) = 1;
B(10,[18 19]) = -1;
B(11,[9 18]) = 1;
B(11,20) = -1;
B(12,[5 20 21]) = 1;
B(13,19) = 1;
B(13,[22 23 24]) = -1;
B(14,23) = 1;
B(14,[25 26]) = -1;
B(15,[7 25]) = 1;
B(15,[27 28]) = -1;
B(16,28) = 1;
B(16,[29 30]) = -1;
B(17,29) = 1;
B(17,[31 32]) = -1;
B(18,32) = 1;
B(18,[33 34]) = -1;
B(19,[3 33]) = 1;
B(19,35) = -1;
B(20,35) = 1;
B(20,[21 36]) = -1;
B(21,36) = 1;
B(21,[37 38]) = -1;
B(22,[30 37]) = 1;
B(22,39) = -1;
B(23,[38 39]) = 1;
B(23,40) = -1;
B(24,40) = 1;
B(24,[41 42]) = -1;
B(25,41) = 1;
B(25,[43 44 50]) = -1;
B(26,44) = 1;
B(26,[45 46 47]) = -1;
B(27,46) = 1;
B(27,48) = -1;
B(28,[42 45 48 50]) = 1;
B(28,49) = -1;
B(29,[26 34 43]) = 1;
B(30,[15 17 24 47]) = 1;
B(31,49) = 1;
B(32,22) = 1;
B(33,27) = 1;
data.A = sparse([eye(50); B]);
clear B
xu = Inf*ones(50,1);
xl = -xu;
x = zeros(50,1);