function croppingImageFiles
inDir = 'F:\個人フォルダ\M1\Uchimi\testImgs\Img_160711\';

files = dir( strcat(inDir, '*.jpg') );
roi = [1366.5 946.5 2544 2324];

for n=1:size(files, 1)
    img = imread( strcat(inDir, files(n).name) );
    crp = imcrop(img, roi);
    imwrite(crp, strcat('./', files(n).name));
end

end