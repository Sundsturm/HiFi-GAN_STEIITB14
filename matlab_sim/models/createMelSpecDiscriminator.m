function lgraph = createMelSpecDiscriminator()
    % Konfigurasi StarGAN-VC / HiFi-GAN Mel Discriminator
    % Input: Mel-Spectrogram [80 x Time x 1]
    
    channelSize = 32; 
    
    % REVISI: Ganti NaN dengan angka nyata. 
    % 32 adalah estimasi lebar Mel-Spec untuk audio 8192 sample (8192 / 256 hop).
    % Angka ini hanya template inisialisasi, network tetap bisa menerima input yang lebih panjang.
    inputSize = [80, 32, 1]; 
    
    lgraph = layerGraph();
    
    % 1. Input Layer (FIXED: Value must be finite)
    lgraph = addLayers(lgraph, imageInputLayer(inputSize, ...
        'Name', 'input', 'Normalization', 'none'));
    
    lastLayerName = 'input';
    
    % Parameter Layer (Kernel & Stride)
    kernels = [3, 9; 3, 8; 3, 8; 3, 6];
    strides = [1, 2; 1, 2; 1, 2; 1, 2];
    
    numStacks = size(kernels, 1);
    
    for i = 1:numStacks
        kFreq = kernels(i, 1); kTime = kernels(i, 2);
        sFreq = strides(i, 1); sTime = strides(i, 2);
        
        blockName = sprintf('stack%d', i);
        
        % --- A. CONV UTAMA (Downsampling) ---
        convMainName = [blockName '_main_conv'];
        bnName = [blockName '_bn'];
        
        lgraph = addLayers(lgraph, convolution2dLayer([kFreq, kTime], channelSize, ...
            'Stride', [sFreq, sTime], 'Padding', 'same', 'Name', convMainName));
        lgraph = addLayers(lgraph, batchNormalizationLayer('Name', bnName));
        
        lgraph = connectLayers(lgraph, lastLayerName, convMainName);
        lgraph = connectLayers(lgraph, convMainName, bnName);
        
        % --- B. GLU BRANCHING ---
        % Cabang 1: Gate (Sigmoid)
        gateConvName = [blockName '_gate_conv'];
        gateActName  = [blockName '_gate_sigmoid'];
        
        lgraph = addLayers(lgraph, convolution2dLayer([3, 3], channelSize, ...
            'Stride', [1, 1], 'Padding', 'same', 'Name', gateConvName));
        lgraph = addLayers(lgraph, sigmoidLayer('Name', gateActName));
        
        % Cabang 2: Info (Linear/Identity)
        infoConvName = [blockName '_info_conv'];
        lgraph = addLayers(lgraph, convolution2dLayer([3, 3], channelSize, ...
            'Stride', [1, 1], 'Padding', 'same', 'Name', infoConvName));
        
        % --- C. MERGING (MULTIPLICATION) ---
        multName = [blockName '_mult'];
        lgraph = addLayers(lgraph, multiplicationLayer(2, 'Name', multName));
        
        % Wiring
        % BN -> Gate
        lgraph = connectLayers(lgraph, bnName, gateConvName);
        lgraph = connectLayers(lgraph, gateConvName, gateActName);
        
        % BN -> Info
        lgraph = connectLayers(lgraph, bnName, infoConvName);
        
        % Connect to Multiplication (in1 & in2)
        lgraph = connectLayers(lgraph, gateActName, [multName '/in1']);
        lgraph = connectLayers(lgraph, infoConvName, [multName '/in2']);
        
        lastLayerName = multName;
    end
    
    % --- Output Conv ---
    % Mengubah menjadi 1 Channel map
    finalConvName = 'final_conv';
    lgraph = addLayers(lgraph, convolution2dLayer([1, 1], 1, 'Padding', 'same', 'Name', finalConvName));
    lgraph = connectLayers(lgraph, lastLayerName, finalConvName);
end