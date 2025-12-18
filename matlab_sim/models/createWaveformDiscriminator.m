% Waveform Discriminator
function lgraph = createWaveformDiscriminator()
    % createWaveformDiscriminator
    % Membangun Waveform Discriminator sesuai konfigurasi MelGAN & Gambar #3
    
    % --- 1. DEFINISI PARAMETER (Sesuai Gambar & Teks) ---
    kernelSizes  = [15, 41, 41, 41, 41, 5, 3];
    strideSizes  = [1,  4,  4,  4,  4, 1, 1];
    channelSizes = [16, 64, 256, 1024, 1024, 1024, 1];
    groupSizes   = [1,  4,  16, 64, 256, 1, 1];
    
    numLayers = length(kernelSizes); % Total 7 Layer
    
    lgraph = layerGraph();
    
    % --- 2. INPUT LAYER ---
    % Input berupa Raw Audio Waveform (1 Channel)
    inputName = 'wav_input';
    lgraph = addLayers(lgraph, sequenceInputLayer(1, 'Name', inputName, 'Normalization', 'none'));
    lastLayerName = inputName;
    
    % --- 3. LOOP KONSTRUKSI 7 LAYER ---
    for i = 1:numLayers
        % Ambil parameter untuk layer ke-i
        k = kernelSizes(i);
        s = strideSizes(i);
        c = channelSizes(i);
        g = groupSizes(i);
        
        layerIdxName = sprintf('conv_%d', i);
        
        % Definisi Grouped Convolution Layer
        % 'Padding same' penting agar dimensi spasial konsisten dengan stride
        convLayer = convolution1dLayer(k, c, ...
            'Stride', s, ...
            'Padding', 'same', ...
            'NumGroups', g, ... % Parameter kunci dari MelGAN
            'Name', layerIdxName);
            
        lgraph = addLayers(lgraph, convLayer);
        lgraph = connectLayers(lgraph, lastLayerName, layerIdxName);
        lastLayerName = layerIdxName;
        
        % Tambahkan Leaky ReLU (Hanya untuk Layer 1 s/d 6)
        % Sesuai gambar, kotak terakhir (Layer 7) hanya bertuliskan "Conv."
        % tanpa kotak "Leaky Relu" di sebelahnya.
        if i < numLayers
            actName = sprintf('lrelu_%d', i);
            % Slope 0.2 adalah standar umum MelGAN
            lgraph = addLayers(lgraph, leakyReluLayer(0.2, 'Name', actName));
            lgraph = connectLayers(lgraph, lastLayerName, actName);
            lastLayerName = actName;
        end
    end
    
    % --- 4. GLOBAL AVERAGE MEAN POOL ---
    % Sesuai kotak paling kanan di gambar: "Global Average Mean Pool"
    poolName = 'global_pool';
    lgraph = addLayers(lgraph, globalAveragePooling1dLayer('Name', poolName));
    lgraph = connectLayers(lgraph, lastLayerName, poolName);
    
    % Output akhir: Skor skalar [1 x 1 x Batch]
end