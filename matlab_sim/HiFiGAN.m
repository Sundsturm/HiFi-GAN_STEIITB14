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

% Downsampling Frequency
function [audio_8k, audio_4k] = downsampleAudio(audio_16k, fs_input)
    % downsampleAudio Mengubah sampling rate audio dengan faktor 2 bertahap.
    %
    % Input:
    %   audio_16k : Sinyal audio asli (Vector / dlarray)
    %   fs_input  : Sampling rate asli (harus 16000 untuk skenario ini)
    %
    % Output:
    %   audio_8k  : Audio hasil downsample ke 8 kHz
    %   audio_4k  : Audio hasil downsample ke 4 kHz
    
    if nargin < 2
        fs_input = 16000;
    end
    
    % Validasi input apakah dlarray (Deep Learning Array) atau Double biasa
    isDlarray = isa(audio_16k, 'dlarray');
    if isDlarray
        % Konversi ke double dulu untuk proses signal processing MATLAB standar
        data_in = extractdata(audio_16k);
        % Pastikan bentuk kolom (Samples x 1)
        if size(data_in, 2) > size(data_in, 1)
            data_in = data_in'; 
        end
    else
        data_in = audio_16k;
    end

    % --- TAHAP 1: 16 kHz ke 8 kHz ---
    % Menggunakan 'resample'. Argumen (1, 2) berarti dikali 1 lalu dibagi 2.
    % Ini menerapkan filter anti-aliasing otomatis.
    audio_8k_raw = resample(data_in, 1, 2);
    
    % --- TAHAP 2: 8 kHz ke 4 kHz ---
    % Inputnya adalah hasil dari tahap 1
    audio_4k_raw = resample(audio_8k_raw, 1, 2);
    
    % --- Format Output ---
    % Jika input awal adalah dlarray, kembalikan sebagai dlarray
    % Biasanya format TCB (Time, Channel, Batch) atau CB (Channel, Batch)
    % Di sini kita kembalikan sebagai vektor kolom standar
    if isDlarray
        audio_8k = dlarray(single(audio_8k_raw));
        audio_4k = dlarray(single(audio_4k_raw));
    else
        audio_8k = audio_8k_raw;
        audio_4k = audio_4k_raw;
    end
    
    % Tampilkan info (Opsional)
    fprintf('Input: %d samples (16 kHz)\n', length(data_in));
    fprintf('Output 1: %d samples (8 kHz)\n', length(audio_8k_raw));
    fprintf('Output 2: %d samples (4 kHz)\n', length(audio_4k_raw));
end

% L1 Sample Loss
function loss = l1SampleLoss(YTrue, YPred)
    % YTrue: Audio Asli (dlarray: Time x 1 x Batch)
    % YPred: Audio Generator (dlarray: Time x 1 x Batch)
    
    % Hitung Mean Absolute Error
    loss = mean(abs(YTrue - YPred), 'all');
end

% Log Spectogram Loss
function totalLoss = dualScaleSpectrogramLoss(dlYTrue, dlYPred)
    % dlYTrue: Ground Truth Audio (dlarray: Time x 1 x Batch)
    % dlYPred: Generated Audio (dlarray: Time x 1 x Batch)
    % Asumsi: Data format adalah CBT (Channel, Batch, Time) atau TB (Time, Batch)
    % Sesuaikan DataFormat di dlstft jika perlu.
    
    % --- KONFIGURASI SESUAI REQUEST ---
    % Kolom 1: FFT/Window Size
    % Kolom 2: Hop Size
    % Baris 1: Large Scale
    % Baris 2: Small Scale
    scales = [2048, 512; ... 
              512,  128];
          
    totalLoss = 0;
    epsilon = 1e-6; % Stabilizer untuk log agar tidak NaN
    
    % Loop untuk kedua skala
    for i = 1:size(scales, 1)
        nfft = scales(i, 1);
        hop = scales(i, 2);
        winLen = nfft; % Biasanya Window Size = FFT Size
        
        % Membuat Hann Window
        % 'periodic' digunakan untuk spectral analysis agar tidak bias
        win = hann(winLen, 'periodic');
        
        % Hitung STFT (Short-Time Fourier Transform)
        % Overlap = WindowLength - HopLength
        overlap = winLen - hop;
        
        % STFT Real Audio
        S_true = dlstft(dlYTrue, 'Window', win, 'OverlapLength', overlap, ...
            'FFTLength', nfft, 'DataFormat', 'CBT'); 
            
        % STFT Generated Audio
        S_pred = dlstft(dlYPred, 'Window', win, 'OverlapLength', overlap, ...
            'FFTLength', nfft, 'DataFormat', 'CBT');
        
        % Ambil Magnitude
        mag_true = abs(S_true);
        mag_pred = abs(S_pred);
        
        % --- PERHITUNGAN LOSS ---
        
        % 1. Log-Magnitude Loss (L1 distance pada log domain)
        % Ini adalah inti dari "Log Spectrogram Loss"
        log_loss = mean(abs(log(mag_true + epsilon) - log(mag_pred + epsilon)), 'all');
        
        % 2. Spectral Convergence Loss (Opsional tapi Standar)
        % Biasanya dipasangkan dengan log loss untuk stabilitas (Frobenius norm metric)
        % Jika Anda HANYA ingin Log Loss murni, Anda bisa menonaktifkan baris ini.
        sc_loss = norm(mag_true - mag_pred, 'fro') / (norm(mag_true, 'fro') + epsilon);
        
        % Akumulasi Loss (Equally Weighted)
        % total = total + (LogLoss + SpectralConvergence)
        totalLoss = totalLoss + log_loss + sc_loss;
    end
    
    % Jika Anda ingin murni rata-rata, bisa dibagi 2. 
    % Tapi dalam loss function, jumlah (sum) atau rata-rata (mean) 
    % seringkali ekuivalen secara gradien (hanya beda scaling rate).
end

function dlMelSpec = stftMelBlock(dlWaveform)
    % stftMelBlock Mengubah Raw Audio menjadi Log Mel-Spectrogram
    % Input:
    %   dlWaveform: Audio [Time x 1 x Batch] (dlarray)
    % Output:
    %   dlMelSpec : Log Mel-Spec [80 x Time x 1 x Batch] (Image-like)

    % --- 1. Konfigurasi Parameter (Standar HiFi-GAN/MelGAN) ---
    fs = 16000;          % Sampling Rate (Sesuai request sebelumnya)
    nfft = 1024;         % FFT Size
    hopLength = 256;     % Hop Size (jarak antar frame)
    winLength = 1024;    % Window Size
    numMels = 80;        % Jumlah Mel Bins (Sesuai parameter Discriminator)
    fMin = 20;           % 20 Hz
    fMax = 8000;         % 8000 Hz
    
    % --- 2. Persiapan (Hanya sekali jalan / Persistent) ---
    persistent melFilterBank hannWin
    
    if isempty(melFilterBank)
        % Membuat Hann Window
        hannWin = hann(winLength, 'periodic');
        
        % Membuat Mel Filter Bank Matrix [NumMels x (NFFT/2 + 1)]
        % Menggunakan fungsi designAuditoryFilterBank (Audio Toolbox)
        % atau kita buat manual agar kompatibel dlarray.
        % Di sini kita gunakan designAuditoryFilterBank jika ada, atau logic simple.
        
        % Opsi A: Menggunakan Audio Toolbox
        melFilterBank = designAuditoryFilterBank(fs, ...
            'FFTLength', nfft, ...
            'NumBands', numMels, ...
            'FrequencyRange', [fMin, fMax], ...
            'Normalization', 'none');
        
        % Konversi ke format single & dlarray agar gpu-ready
        melFilterBank = dlarray(single(melFilterBank)); 
    end
    
    % --- 3. Proses STFT (Differentiable) ---
    % Input dlWaveform biasanya [Time, 1, Batch]
    % dlstft mengharapkan input format CBT atau TB
    
    % Hitung STFT
    % Output: [FreqBins, Time, Channel, Batch]
    % FreqBins = (nfft/2) + 1 = 513
    dlSpecComplex = dlstft(dlWaveform, ...
        'Window', hannWin, ...
        'OverlapLength', winLength - hopLength, ...
        'FFTLength', nfft, ...
        'DataFormat', 'TCB'); % Sesuaikan format input Anda
        
    % Ambil Magnitude (Absolut)
    dlSpecMag = abs(dlSpecComplex);
    
    % --- 4. Mel-Filter Bank Matrix Multiplication ---
    % Mel Matrix: [80 x 513]
    % Spec Mag  : [513 x Time x 1 x Batch]
    % Kita perlu melakukan perkalian matriks: (Mel x Spec)
    
    % Reshape untuk perkalian matriks batch (collapse Channel & Batch)
    [nFreq, nTime, nCh, nBatch] = size(dlSpecMag);
    
    % Flatten: [513, Time * Batch]
    X_flat = reshape(dlSpecMag, nFreq, []);
    
    % Perkalian: [80 x 513] * [513 x N] -> [80 x N]
    melFlat = melFilterBank * X_flat;
    
    % Reshape kembali ke [80, Time, 1, Batch]
    dlMelSpecLinear = reshape(melFlat, numMels, nTime, nCh, nBatch);
    
    % --- 5. Logarithm (Dynamic Range Compression) ---
    epsilon = 1e-5; % Mencegah log(0)
    dlMelSpec = log(dlMelSpecLinear + epsilon);
    
    % Output kini berdimensi [80, Time, 1, Batch]
    % Siap masuk ke Mel-Spectrogram Discriminator (Conv2D)
end