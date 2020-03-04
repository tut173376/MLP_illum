%% image resize (max:1024pix)

function im = resizeImg(img)

rSize = [NaN, NaN];
[ M, I ] = max(size(img));
if M > 1024 + 100
    rSize(I) = 1024;
    im = imresize( img, rSize );
else
    im = img;
end

im = max(0,min(1,im));