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