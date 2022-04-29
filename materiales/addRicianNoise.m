function [ outputImage ] = addRicianNoise( inputImage, noiseIntensity)
%addRicianNoise Adds MRI noise to an image
%   inputImage is in double format
%   noiseIntensity is the relation between the average image intensity and
%   de average noise intensity


if nargin < 2

    noiseIntensity = 0.2;

end

pd = makedist('Rician');

random = rand(size(inputImage));

noise = icdf(pd, random);

meanNoise = mean(noise(:));

meanSignal = mean(inputImage(:));

noiseFactor = noiseIntensity * meanSignal / meanNoise;

outputImage = inputImage + noiseFactor .* noise;

end

