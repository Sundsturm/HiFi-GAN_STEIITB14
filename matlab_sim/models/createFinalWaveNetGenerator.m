% WaveNet Generator
function lgraph = createFinalWaveNetGenerator()
    % "Two stacks of dilated convolutions"
    numStacks = 2;              
    
    % "Dilation rates from 1 to 512"
    % 2^0 = 1, ..., 2^9 = 512. (Total 10 layers per stack)
    numLayersPerStack = 10;     
    
    % "Filter size of 3"
    filterSize = 3;             
    
    % "Channel size is 128 across the network"
    channelSize = 128;          
    
    inputChannels = 1; % Audio Mono
    
    % --- MEMULAI GRAPH ---
    lgraph = layerGraph();
    
    % 1. Input Layer
    lgraph = addLayers(lgraph, sequenceInputLayer(inputChannels, 'Name', 'input', 'Normalization', 'none'));
    
    % 2. Initial Projection (Memastikan channel masuk ke 128)
    initConvName = 'init_conv';
    % Menggunakan filterSize 3 untuk proyeksi awal (standar WaveNet)
    lgraph = addLayers(lgraph, convolution1dLayer(filterSize, channelSize, ...
        'Padding', 'same', 'Name', initConvName));
    lgraph = connectLayers(lgraph, 'input', initConvName);
    
    lastLayerName = initConvName;
    
    % --- CORE LOOP: 2 STACKS x 10 LAYERS ---
    for s = 1:numStacks
        for i = 0:(numLayersPerStack - 1)
            
            % Menghitung Dilasi: 1, 2, 4, 8, ..., 512
            dilation = 2^i;
            
            % Penamaan unik untuk setiap layer
            baseName = sprintf('stack%d_dil%d', s, dilation);
            
            % --- A. GATED ATTENTION UNIT (Filter size 3) ---
            
            % Cabang 1: CONTENT (Tanh) - Filter Size 3
            filtConvName = [baseName '_filt_conv'];
            filtActName = [baseName '_filt_tanh'];
            
            lgraph = addLayers(lgraph, convolution1dLayer(filterSize, channelSize, ...
                'DilationFactor', dilation, 'Padding', 'same', 'Name', filtConvName));
            lgraph = addLayers(lgraph, tanhLayer('Name', filtActName));
            
            % Cabang 2: GATE (Sigmoid) - Filter Size 3
            gateConvName = [baseName '_gate_conv'];
            gateActName = [baseName '_gate_sigm'];
            
            lgraph = addLayers(lgraph, convolution1dLayer(filterSize, channelSize, ...
                'DilationFactor', dilation, 'Padding', 'same', 'Name', gateConvName));
            lgraph = addLayers(lgraph, sigmoidLayer('Name', gateActName));
            
            % Cabang 3: MULTIPLICATION (Element-wise)
            multName = [baseName '_gated_mult'];
            lgraph = addLayers(lgraph, multiplicationLayer(2, 'Name', multName));
            
            % --- WIRING GATED UNIT ---
            % Input -> Filter branch
            lgraph = connectLayers(lgraph, lastLayerName, filtConvName);
            lgraph = connectLayers(lgraph, filtConvName, filtActName);
            
            % Input -> Gate branch
            lgraph = connectLayers(lgraph, lastLayerName, gateConvName);
            lgraph = connectLayers(lgraph, gateConvName, gateActName);
            
            % Merge branches
            lgraph = connectLayers(lgraph, filtActName, [multName '/1']);
            lgraph = connectLayers(lgraph, gateActName, [multName '/2']);
            
            % --- B. RESIDUAL CONNECTION (Channel Mixing) ---
            % 1x1 Convolution wajib digunakan di sini untuk mencampur fitur
            % sebelum dijumlahkan kembali (Residual).
            mixName = [baseName '_1x1_proj'];
            lgraph = addLayers(lgraph, convolution1dLayer(1, channelSize, ...
                'Padding', 'same', 'Name', mixName));
            lgraph = connectLayers(lgraph, multName, mixName);
            
            % Addition Layer (Input Awal + Hasil Block)
            addName = [baseName '_res_add'];
            lgraph = addLayers(lgraph, additionLayer(2, 'Name', addName));
            
            % Connect Mixing Output -> Add
            lgraph = connectLayers(lgraph, mixName, [addName '/1']);
            
            % Connect Skip/Residual Path (Input Awal) -> Add
            lgraph = connectLayers(lgraph, lastLayerName, [addName '/2']);
            
            % Update pointer layer terakhir
            lastLayerName = addName;
        end
    end
    
    % --- 3. FINAL PROJECTION ---
    % Mengubah channel 128 kembali menjadi 1 (Audio Output)
    
    postConvName = 'post_conv';
    lgraph = addLayers(lgraph, convolution1dLayer(1, channelSize, 'Padding', 'same', 'Name', postConvName));
    lgraph = addLayers(lgraph, leakyReluLayer(0.2, 'Name', 'post_relu'));
    
    lgraph = connectLayers(lgraph, lastLayerName, postConvName);
    lgraph = connectLayers(lgraph, postConvName, 'post_relu');
    
    finalConvName = 'final_conv';
    lgraph = addLayers(lgraph, convolution1dLayer(1, 1, 'Padding', 'same', 'Name', finalConvName));
    % Tanh digunakan di akhir karena audio waveform normalnya -1 s/d 1
    lgraph = addLayers(lgraph, tanhLayer('Name', 'audio_out'));
    
    lgraph = connectLayers(lgraph, 'post_relu', finalConvName);
    lgraph = connectLayers(lgraph, finalConvName, 'audio_out');
end