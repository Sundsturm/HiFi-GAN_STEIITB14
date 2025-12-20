function [grads, totalLoss] = modelGradients_Stage1(netG, X, Y)
    % Forward Pass
    Y_Pred = forward(netG, X);
    
    % --- Hitung Losses ---
    
    % 1. L1 Loss (Time Domain)
    % Pastikan input real
    lossL1 = l1SampleLoss(real(Y), real(Y_Pred));
    
    % 2. Spectrogram Loss (Freq Domain)
    % Fungsi ini sudah kita update untuk handle 'real' di dalamnya
    lossSpec = dualScaleSpectrogramLoss(Y, Y_Pred);
    
    % --- Total Loss ---
    % FIX: Gunakan real() pada penjumlahan akhir untuk membuang 
    % sisa-sisa imajiner (numerical epsilon)
    totalLoss = real(lossL1 + lossSpec);
    
    % Hitung Gradient
    grads = dlgradient(totalLoss, netG.Learnables);
end