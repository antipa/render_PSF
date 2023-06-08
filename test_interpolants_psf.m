px = [-1, 1.1, 1.89, .2, -1.6, 0]';
py = [-1.3, -.9, 2, 2.1, 1.3,0]'
P = cat(2,px,py);
V = [0; 0; 0; 0; 0; 1];
scatter3(px,py,V)
%%
F = scatteredInterpolant(P,V)
[X,Y] = meshgrid(linspace(-2,2,50), linspace(-2,2,50));
figure(1)
clf
F.Method = 'linear'

V2lin = reshape(F([X(:),Y(:)]), 50,50)
surf(X,Y,V2lin)
title('linear')
hold on
scatter3(px,py,V,200,'r.')
hold off

% vs
figure(2)
F.Method = 'natural'
V2nat = reshape(F([X(:),Y(:)]), 50,50)
surf(X,Y,V2nat)
title('natural')
hold on
scatter3(px,py,V,200,'r.')