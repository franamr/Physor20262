
hp = 0.02;
Point(1) = {0, 0, 0, hp};
Point(2) = {0,0.1, 0, hp};
Point(3) = {0, 0.7, 0, hp};
Point(4) = {0, 0.9, 0, hp};
Point(5) = {0, 1.3, 0, hp};
Point(6) = {0, 1.5, 0, hp};
Point(7) = {0, 1.7, 0, hp};

Point(8) = {0.3, 0.3, 0, hp};
Point(9) = {0.3, 0.5, 0, hp};
Point(10) = {0.3, 1.1, 0, hp};
Point(11) = {0.3, 1.3, 0, hp};

Point(12) = {0.5, 0.3, 0, hp};
Point(13) = {0.5, 0.5, 0, hp};
Point(14) = {0.5, 1.3, 0, hp};
Point(15) = {0.5, 1.5, 0, hp};

Point(16) = {0.7, 0, 0, hp};
Point(17) = {0.7, 0.1, 0, hp};
Point(18) = {0.7, 0.7, 0, hp};
Point(19) = {0.7, 0.9, 0, hp};
Point(20) = {0.7, 1.1, 0, hp};
Point(21) = {0.7, 1.5, 0, hp};
Point(22) = {0.7, 1.7, 0, hp};

Point(23) = {0.9, 0, 0, hp};
Point(24) = {0.9, 0.1, 0, hp};
Point(25) = {0.9, 0.7, 0, hp};
Point(26) = {0.9, 0.9, 0, hp};
Point(27) = {0.9, 1.1, 0, hp};
Point(28) = {0.9, 1.3, 0, hp};

Point(29) = {1.1, 0.3, 0, hp};
Point(30) = {1.1, 0.7, 0, hp};
Point(31) = {1.1, 0.9, 0, hp};
Point(32) = {1.1, 1.1, 0, hp};
Point(33) = {1.1, 1.3, 0, hp};
Point(34) = {1.1, 1.5, 0, hp};

Point(35) = {1.3, 0, 0, hp};
Point(36) = {1.3, 0.3, 0, hp};
Point(37) = {1.3, 0.5, 0, hp};
Point(38) = {1.3, 0.9, 0, hp};
Point(39) = {1.3, 1.1, 0, hp};
Point(40) = {1.3, 1.3, 0, hp};

Point(41) = {1.5, 0, 0, hp};
Point(42) = {1.5, 0.5, 0, hp};
Point(43) = {1.5, 0.7, 0, hp};
Point(44) = {1.5, 1.1, 0, hp};

Point(45) = {1.7, 0, 0, hp};
Point(46) = {1.7, 0.7, 0, hp};

Point(47) = {0.1, 0, 0, hp};
Point(48) = {0.1, 0.1, 0 ,hp};
Point(49) = {0.1, 0.7, 0, hp};
Point(50) = {0.1, 0.9, 0 ,hp};

Line(1) = {6, 7};
Line(2) = {7, 22};
Line(3) = {22, 21};
Line(4) = {21, 34};
Line(5) = {34, 33};
Line(6) = {33, 40};
Line(7) = {40, 39};
Line(8) = {39, 44};
Line(9) = {44, 43};
Line(10) = {43, 46};
Line(11) = {46, 45};
Line(12) = {45, 41};
Line(13) = {41, 42};
Line(14) = {42, 37};
Line(15) = {37, 38};
Line(16) = {38, 31};
Line(17) = {31, 32};
Line(18) = {32, 27};
Line(19) = {27, 28};
Line(20) = {28, 14};
Line(21) = {14, 15};
Line(22) = {15, 6};

Line(23) = {6, 5};
Line(24) = {5, 11};
Line(25) = {11, 10};
Line(26) = {10, 20};
Line(27) = {20, 19};
Line(28) = {19, 26};
Line(29) = {26, 25};
Line(30) = {25, 30};
Line(31) = {30, 29};
Line(32) = {29, 36};
Line(33) = {36, 35};
Line(34) = {35, 41};

Line(35) = {35, 23};
Line(36) = {16, 47};
Line(37) = {2, 3};
Line(38) = {4, 5};

Line(39) = {18, 25};
Line(40) = {18, 19};
Line(41) = {3, 4};
Line(42) = {4, 50};
Line(43) = {50, 49};
Line(44) = {3, 49};
Line(45) = {8, 9};
Line(46) = {9, 13};
Line(47) = {12, 13};
Line(48) = {8, 12};

Line(49) = {16, 23};
Line(50) = {23, 24};
Line(51) = {24, 17};
Line(52) = {16, 17};

Line(53) = {1, 47};
Line(54) = {47, 48};
Line(55) = {48, 2};
Line(56) = {2, 1};

Line Loop(57) = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22};
Line Loop(58) = {22,21,20,19,18,17,16,15,14,13,34,33,32,31,30,29,28,27,26,25,24,23};
Line Loop(59) = {24,25,26,27,-40,39,30,31,32,33,35,50,51,-52,36,54,55,37,44,-43,-42,38};

Line Loop(60) = {28,29,-39,40};
Line Loop(61) = {41,42,43,-44};
Line Loop(62) = {45,46,-47,-48};
Line Loop(63) = {49,50,51,-52};
Line Loop(64) = {53,54,55,56};


Plane Surface(101) = {57};
Plane Surface(102) = {58};
Plane Surface(103) = {59, 62};

Plane Surface(104) = {60};
Plane Surface(105) = {61};
Plane Surface(106) = {62};
Plane Surface(107) = {63};
Plane Surface(108) = {64};


Physical Surface(1) = {101};
Physical Surface(2) = {102};
Physical Surface(3) = {103};
Physical Surface(4) = {104};
Physical Surface(5) = {105};
Physical Surface(6) = {106};
Physical Surface(7) = {107};
Physical Surface(8) = {108};

Physical Line(11) = {2,3,4,5,6,7,8,9,10,11};
Physical Line(12) = {13,14,15,16,17,18,19,20,21.22};

// --- Interfaces / segmentos interiores (ejemplos) ---
// agujero/contorno pequeño de loop 62 (8-9-13-12)
Physical Curve(13) = {45,46,47,48};

// rectángulo interno de loop 63 (16-17-24-23)
Physical Curve(14) = {49,50,51,52};

// “slot”/contorno de loop 60 (28-29-39-40)
Physical Curve(15) = {28,29,39,40};

// contorno de loop 61 (41-42-43-44)
Physical Curve(16) = {41,42,43,44};


Physical Curve(17) = {24,25,26,27,28,29,30,21,23,33,34};

