function [gradsD, lossD_Total] = gradients_DiscriminatorOnlyStage3(netG, netP, D1, D2, D3, DM, X, Y)
    % 1. Generate Fake Audio
    Y_Mid = forward(netG, X);
    Y_Fake = forward(netP, Y_Mid);
    Y_Real = Y;
    
    % 2. Siapkan Multi-Scale Inputs
    % Scale 1 (16k)
    R1 = Y_Real; F1 = Y_Fake;
    % Scale 2 (8k)
    [R2, ~] = downsampleAudio(Y_Real, 16000);
    [F2, ~] = downsampleAudio(Y_Fake, 16000);
    % Scale 3 (4k)
    [~, R3] = downsampleAudio(Y_Real, 16000);
    [~, F3] = downsampleAudio(Y_Fake, 16000);
    % Mel-Spec
    R_Mel = stftMelBlock(Y_Real);
    F_Mel = stftMelBlock(Y_Fake);
    
    % 3. Forward & Hitung Hinge Loss (Hanya Loss D yang dipakai)
    
    % --- Waveform Disc 1 ---
    score_R1 = forward(D1, R1); score_F1 = forward(D1, F1);
    [~, lossD1] = hingeGANLoss(score_R1, score_F1);
    
    % --- Waveform Disc 2 ---
    score_R2 = forward(D2, R2); score_F2 = forward(D2, F2);
    [~, lossD2] = hingeGANLoss(score_R2, score_F2);
    
    % --- Waveform Disc 3 ---
    score_R3 = forward(D3, R3); score_F3 = forward(D3, F3);
    [~, lossD3] = hingeGANLoss(score_R3, score_F3);
    
    % --- Mel Disc ---
    score_RM = forward(DM, R_Mel); score_FM = forward(DM, F_Mel);
    [~, lossDM] = hingeGANLoss(score_RM, score_FM);
    
    % Total Loss Discriminator
    lossD_Total = lossD1 + lossD2 + lossD3 + lossDM;
    
    % 4. Hitung Gradien
    gradsD.D1 = dlgradient(lossD1, D1.Learnables);
    gradsD.D2 = dlgradient(lossD2, D2.Learnables);
    gradsD.D3 = dlgradient(lossD3, D3.Learnables);
    gradsD.DM = dlgradient(lossDM, DM.Learnables);
end