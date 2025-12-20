function [gradsD, lossD_Total] = gradients_DiscriminatorOnlyStage3(netG, netP, D1, D2, D3, DM, X, Y)
    % 1. Generate Fake Audio
    Y_Mid = forward(netG, X);
    Y_Fake = forward(netP, Y_Mid);
    Y_Real = Y;
    
    % Force Real
    Y_Fake = real(Y_Fake);
    Y_Real = real(Y_Real);
    
    % 2. Siapkan Multi-Scale Inputs (DENGAN TARGET SR YANG BENAR)
    
    % Scale 1: Original (16000 Hz)
    R1 = Y_Real; 
    F1 = Y_Fake;
    
    % Scale 2: Downsample 2x (8000 Hz)
    % PERBAIKAN: Ambil output PERTAMA [R2, ~], bukan [~, R2]
    [R2, ~] = downsampleAudio(Y_Real, 8000);
    [F2, ~] = downsampleAudio(Y_Fake, 8000);
    
    % Scale 3: Downsample 4x (4000 Hz)
    % PERBAIKAN: Ambil output PERTAMA [R3, ~]
    [R3, ~] = downsampleAudio(Y_Real, 4000);
    [F3, ~] = downsampleAudio(Y_Fake, 4000);
    
    % Mel-Spec
    R_Mel = stftMelBlock(Y_Real);
    F_Mel = stftMelBlock(Y_Fake);
    
    % 3. Forward & Hinge Loss
    
    % D1 (Original)
    score_R1 = forward(D1, R1); score_F1 = forward(D1, F1);
    [~, lossD1] = hingeGANLoss(score_R1, score_F1);
    
    % D2 (Downsampled)
    score_R2 = forward(D2, R2); score_F2 = forward(D2, F2);
    [~, lossD2] = hingeGANLoss(score_R2, score_F2);
    
    % D3 (Downsampled More)
    score_R3 = forward(D3, R3); score_F3 = forward(D3, F3);
    [~, lossD3] = hingeGANLoss(score_R3, score_F3);
    
    % D_Mel
    score_RM = forward(DM, R_Mel); score_FM = forward(DM, F_Mel);
    [~, lossDM] = hingeGANLoss(score_RM, score_FM);
    
    % Total Loss
    lossD_Total = lossD1 + lossD2 + lossD3 + lossDM;
    
    % 4. Gradients
    % Karena lossD1 hanya bergantung pada D1, lossD2 pada D2, dst..
    % dlgradient bisa menghitungnya secara terpisah dengan aman.
    gradsD.D1 = dlgradient(lossD1, D1.Learnables);
    gradsD.D2 = dlgradient(lossD2, D2.Learnables);
    gradsD.D3 = dlgradient(lossD3, D3.Learnables);
    gradsD.DM = dlgradient(lossDM, DM.Learnables);
end