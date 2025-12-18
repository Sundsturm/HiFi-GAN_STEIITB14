% PostNet
function lgraph = createSpecificPostNet(inputChannels)
    % createSpecificPostNet Membangun PostNet sesuai spesifikasi pengguna
    %
    % Argumen:
    %   inputChannels - Jumlah channel input (misal: 80 untuk Mel-Spectrogram 
    %                   atau 1 untuk Waveform, atau 128 jika fitur internal).
    
    if nargin < 1
        inputChannels = 128; % Default asumsi jika tidak ditentukan
    end

    % --- PARAMETER UTAMA ---
    numLayers = 12;         % "12 pasang"
    kernelSize = 32;        % "Kernel length 32"
    numFilters = 128;       % "Channel size 128"
    
    lgraph = layerGraph();
    
    % 1. Input Layer
    inputName = 'postnet_input';
    % Sequence Input: [Channels x Time x Batch]
    lgraph = addLayers(lgraph, sequenceInputLayer(inputChannels, 'Name', inputName, 'Normalization', 'none'));
    
    lastLayerName = inputName;
    
    % 2. Loop Konstruksi 12 Pasang (Conv + Tanh)
    for i = 1:numLayers
        % Nama unik untuk setiap layer agar tidak bentrok di graph
        convName = sprintf('pn_conv_%d', i);
        tanhName = sprintf('pn_tanh_%d', i);
        
        % Definisi Convolution Layer
        % Padding 'same' menjaga panjang waktu (Time steps) tetap sama.
        convLayer = convolution1dLayer(kernelSize, numFilters, ...
            'Stride', 1, ...
            'Padding', 'same', ... 
            'Name', convName);
            
        % Definisi Tanh Layer
        tanhBlock = tanhLayer('Name', tanhName);
        
        % Menambahkan layer ke graph
        lgraph = addLayers(lgraph, convLayer);
        lgraph = addLayers(lgraph, tanhBlock);
        
        % Menghubungkan: Layer Sebelumnya -> Conv -> Tanh
        lgraph = connectLayers(lgraph, lastLayerName, convName);
        lgraph = connectLayers(lgraph, convName, tanhName);
        
        % Update pointer untuk iterasi berikutnya
        lastLayerName = tanhName;
    end
    
    % Output dari fungsi ini adalah layerGraph yang berakhir di Tanh ke-12.
    % Output shape: [128, Time, Batch]
end