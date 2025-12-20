function totalLoss = dualScaleSpectrogramLoss(YTrue, YPred)
    % dualScaleSpectrogramLoss Menghitung Multi-Resolution STFT Loss
    
    % --- FIX: FORCE REAL INPUTS ---
    % Memastikan tidak ada noise imajiner sebelum masuk STFT
    % Ini mencegah error "Complex value in dlconv" saat backprop
    YTrue = real(YTrue);
    YPred = real(YPred);
    
    totalLoss = 0;
    
    % Konfigurasi Multi-Scale
    fftSizes   = [1024, 2048, 512];
    winSizes   = [600,  1200, 240];
    hopSizes   = [120,  240,  50];
    
    numScales = length(fftSizes);
    
    for i = 1:numScales
        n_fft = fftSizes(i);
        win_len = winSizes(i);
        hop_len = hopSizes(i);
        
        win = hann(win_len, 'periodic');
        
        % Hitung STFT (Tanpa 'DataFormat' karena input sudah TCB)
        S_true = dlstft(YTrue, 'Window', win, 'OverlapLength', win_len - hop_len, 'FFTLength', n_fft);
        S_pred = dlstft(YPred, 'Window', win, 'OverlapLength', win_len - hop_len, 'FFTLength', n_fft);
        
        % Magnitude (Complex -> Real)
        Mag_true = abs(S_true);
        Mag_pred = abs(S_pred);
        
        % L1 Convergence Loss
        loss_lin = mean(abs(Mag_true - Mag_pred), 'all');
        
        % Log-Magnitude Loss
        eps_val = 1e-6;
        loss_log = mean(abs(log(Mag_true + eps_val) - log(Mag_pred + eps_val)), 'all');
        
        totalLoss = totalLoss + (loss_lin + loss_log);
    end
    
    totalLoss = totalLoss / numScales;
    
    % --- FIX: FORCE REAL LOSS ---
    % Pastikan output skalar loss benar-benar Real
    totalLoss = real(totalLoss);
end