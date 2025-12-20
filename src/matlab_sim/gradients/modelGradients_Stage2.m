function [grads, totalLoss] = modelGradients_Stage2(netG, netP, X, Y)
    % 1. Forward Pass
    Y_Mid = forward(netG, X);
    Y_Final = forward(netP, Y_Mid);
    
    % 2. Hitung Loss (Pastikan Input REAL untuk memutus rantai kompleks STFT)
    % Kita pakai real() agar dlconv tidak error "Complex value"
    
    % --- Intermediate Loss (Output Generator) ---
    lossL1_Mid   = l1SampleLoss(real(Y), real(Y_Mid));
    lossSpec_Mid = dualScaleSpectrogramLoss(Y, Y_Mid); % Fungsi ini sudah handle real() di dalamnya
    
    % --- Final Loss (Output PostNet) ---
    lossL1_Final   = l1SampleLoss(real(Y), real(Y_Final));
    lossSpec_Final = dualScaleSpectrogramLoss(Y, Y_Final);
    
    % 3. Total Loss
    % Paksa totalLoss menjadi Real sepenuhnya
    totalLoss = real(lossL1_Mid + lossSpec_Mid + lossL1_Final + lossSpec_Final);
    
    % 4. Hitung Gradient Gabungan
    learnables = [netG.Learnables; netP.Learnables];
    gradsAll = dlgradient(totalLoss, learnables);
    
    % 5. Pisahkan Gradient untuk G dan P
    % netG ada di bagian awal, netP di bagian akhir array learnables
    numGParams = size(netG.Learnables, 1);
    
    grads.G = gradsAll(1:numGParams, :);
    grads.P = gradsAll(numGParams+1:end, :);
end