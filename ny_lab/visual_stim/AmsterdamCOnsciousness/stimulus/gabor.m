%Generate 2D Gabor Patch
%
% GABOR
%   Generates a spatially oriented sinusoidal grating multiplied by a gaussian
%   window. Gabor functions have been used to model simple cell receptive fields
%   and are commonly used filters for edge detection.
%
% USAGE
%   m = gabor(varargin)
%
%   Returns:
%       m = width x height matrix (0 to 256)
%
%   Parameters (Default):
%       theta (0): orientation of the gabor patch in degrees
%       freq (0.01): spatial frequency of grating (cycle/degree)
%       sigma (50): standard deviation of gaussian window
%       width (1024): width of generated image
%       height (1024): height of generated image
%       px (1): horizontal center of gabor patch in pixels
%       py (1): vertical center of gabor patch in pixels
%       phase(0): phase offset of grating 0 --> 1
% EXAMPLE
%   m = gabor;
%   imshow(gabor); colormap gray;
%


function m = gabor(varargin)

    % Parse Input
    p = inputParser;
    addParamValue(p,'theta',0,@isnumeric);
    addParamValue(p,'freq',0.01,@isnumeric);
    addParamValue(p,'sigma',50,@isnumeric);
    addParamValue(p,'width',1024,@isnumeric);
    addParamValue(p,'height',1024,@isnumeric);
    addParamValue(p,'px',1,@isnumeric);
    addParamValue(p,'py',1,@isnumeric);
    addParamValue(p,'phase',0,@isnumeric);
    addParamValue(p,'contrast',100,@isnumeric);
    p.KeepUnmatched = true;
    parse(p,varargin{:});
    
    % Compute the gabor patch
    res = 1*[p.Results.width p.Results.height];
                                    
    [gab_x, gab_y] = meshgrid(0:(res(1)-1), 0:(res(2)-1));
    a=cos(deg2rad(p.Results.theta))*p.Results.freq*360;
    b=sin(deg2rad(p.Results.theta))*p.Results.freq*360;
    multConst=1/(sqrt(2*pi)*p.Results.sigma);
    x_factor=-1*(gab_x-p.Results.px).^2;
    y_factor=-1*(gab_y-p.Results.py).^2;
    sinWave=sin(deg2rad(a*(gab_x - p.Results.px) + b*(gab_y - p.Results.py)+p.Results.phase));
    varp.sigmaale=2*p.Results.sigma^2;
    gauss = multConst*exp(x_factor/varp.sigmaale+y_factor/varp.sigmaale);
    m=im2uint8(0.5 + p.Results.contrast*(gauss.*sinWave));    
end