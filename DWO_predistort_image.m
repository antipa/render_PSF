im_in = imread('cameraman.tif');

L = double(im_in);

imagesc(L), axis image

[M,N] = size(L);

x = linspace(-sqrt(2)/2,sqrt(2)/2,N);

[X,Y] = meshgrid(x,x);

h = sqrt(X.^2 + Y.^2);
th = atan2(Y,X);

W311 = -.5;

h_dist = W311*h.^3;

X_dist = (h + h_dist).*cos(th);
Y_dist = (h + h_dist).*sin(th);

L_dist = interp2(X,Y,L,X_dist,Y_dist);
imagesc(L_dist), axis image