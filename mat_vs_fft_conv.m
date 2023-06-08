

tf = 0;
tm = 0;

m = 100;
n = 100;

vec = @(x)x(:);
Ntrial = 100;
for n = 1:Ntrial
    kern= rand(m,n);
    obj = rand(m,n);

    tic;
    K = fft2(kern);
    O = fft2(obj);
    ko = ifft2(K.*O);
    tf = tf + toc;

    tic; 
    Kmat = toeplitz(vec(kern),vec(kern));
    ko = reshape(Kmat*vec(obj),[m,n]);
    tm = tm + toc;

end


