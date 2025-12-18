function lgraph = createMelSpecDiscriminator()
    % createMelSpecDiscriminator
    % Membangun Discriminator 2D untuk Mel-Spectrogram (80 bins)
    % Arsitektur: 4 Stacks of Gated CNN -> Conv -> Global Avg Pool
    
    % --- KONFIGURASI PARAMETER (Sesuai Prompt & StarGAN-VC) ---
    % Kernel Sizes: (Freq, Time)
    kernels = [3, 9; ...
               3, 8; ...
               3, 8; ...
               3, 6];
           
    % Stride Sizes: (Freq, Time)
    % Stride (1,2) berarti Frekuensi tetap, Waktu di-downsample setengah.
    strides = [1, 2; ...
               1, 2; ...
               1, 2; ...
               1, 2];
           
    channelSize = 32; % "channel sizes of 32 across the layers"
    inputShape = [80, 128, 1]; % [Freq, Time, Channel]. Time=128 hanyalah placeholder.
    
    lgraph = layerGraph();
    
    % 1. INPUT LAYER
    inputName = 'mel_input';
    % Normalization 'none' karena biasanya Log-Mel sudah dinormalisasi manual
    lgraph = addLayers(lgraph, imageInputLayer(inputShape, 'Name', inputName, 'Normalization', 'none'));
    lastLayerName = inputName;
    
    % 2. LOOP 4 STACK (CUSTOM GATED LAYERS)
    numStacks = 4;
    
    for i = 1:numStacks
        % Parameter untuk layer ini
        kFreq = kernels(i, 1); kTime = kernels(i, 2);
        sFreq = strides(i, 1); sTime = strides(i, 2);
        
        blockName = sprintf('stack%d', i);
        
        % --- A. PRE-PROCESSING (Conv & BN) ---
        % Ini adalah konvolusi utama yang melakukan downsampling
        convMainName = [blockName '_main_conv'];
        bnName = [blockName '_bn'];
        
        lgraph = addLayers(lgraph, convolution2dLayer([kFreq, kTime], channelSize, ...
            'Stride', [sFreq, sTime], ...
            'Padding', 'same', ...
            'Name', convMainName));
            
        lgraph = addLayers(lgraph, batchNormalizationLayer('Name', bnName));
        
        lgraph = connectLayers(lgraph, lastLayerName, convMainName);
        lgraph = connectLayers(lgraph, convMainName, bnName);
        
        % --- B. BRANCHING (GATED LINEAR UNIT LOGIC) ---
        % Output BN dibagi ke dua cabang.
        % Karena parameter kernel cabang tidak dispesifikasikan, kita gunakan 
        % Conv 3x3 (Padding same) untuk mempertahankan dimensi spasial 
        % dan hanya memproses fitur (Standard Practice di GLU Blocks).
        
        % CABANG 1: GATE (Conv + Sigmoid)
        gateConvName = [blockName '_gate_conv'];
        gateActName  = [blockName '_gate_sigmoid'];
        
        lgraph = addLayers(lgraph, convolution2dLayer([3, 3], channelSize, ...
            'Stride', [1, 1], 'Padding', 'same', 'Name', gateConvName));
        lgraph = addLayers(lgraph, sigmoidLayer('Name', gateActName));
        
        lgraph = connectLayers(lgraph, bnName, gateConvName);
        lgraph = connectLayers(lgraph, gateConvName, gateActName);
        
        % CABANG 2: INFO (Conv only)
        infoConvName = [blockName '_info_conv'];
        
        lgraph = addLayers(lgraph, convolution2dLayer([3, 3], channelSize, ...
            'Stride', [1, 1], 'Padding', 'same', 'Name', infoConvName));
            
        lgraph = connectLayers(lgraph, bnName, infoConvName);
        
        % --- C. MERGING (Multiplication) ---
        multName = [blockName '_mult'];
        lgraph = addLayers(lgraph, multiplicationLayer(2, 'Name', multName));
        
        % Sambungkan Gate (Sigmoid) ke input 1
        lgraph = connectLayers(lgraph, gateActName, [multName '/1']);
        % Sambungkan Info (Conv) ke input 2
        lgraph = connectLayers(lgraph, infoConvName, [multName '/2']);
        
        % Update pointer layer terakhir
        lastLayerName = multName;
    end
    
    % 3. FINAL CONVOLUTION LAYER
    % "1 buah Convolution Layer"
    % Biasanya mengubah 32 channel menjadi 1 channel (Score Map)
    finalConvName = 'final_conv';
    lgraph = addLayers(lgraph, convolution2dLayer([3, 3], 1, ...
        'Stride', [1, 1], 'Padding', 'same', 'Name', finalConvName));
    lgraph = connectLayers(lgraph, lastLayerName, finalConvName);
    
    % 4. GLOBAL AVERAGE MEAN POOL
    % "1 buah average global mean pool"
    poolName = 'global_avg_pool';
    lgraph = addLayers(lgraph, globalAveragePooling2dLayer('Name', poolName));
    lgraph = connectLayers(lgraph, finalConvName, poolName);
    
    % Opsional: Flatten layer agar output benar-benar skalar [Batch x 1]
    % lgraph = addLayers(lgraph, flattenLayer('Name', 'flatten'));
    % lgraph = connectLayers(lgraph, poolName, 'flatten');
    
    % Visualisasi Graph (Opsional, hilangkan titik koma untuk melihat)
    % plot(lgraph);
end