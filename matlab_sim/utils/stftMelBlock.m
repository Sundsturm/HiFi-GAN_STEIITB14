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