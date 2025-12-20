function [gradsG, lossG_Total] = gradients_GeneratorOnlyStage3(netG, netP, D1, D2, D3, DM, X, Y)
    % 1. Generate Fake Audio
    Y_Mid = forward(netG, X);
    Y_Fake = forward(netP, Y_Mid);
    Y_Real = Y;
    
    % 2. Reconstruction Losses (L1 + Mel)
    lossRecon = l1SampleLoss(Y_Real, Y_Fake) + dualScaleSpectrogramLoss(Y_Real, Y_Fake);
    
    % 3. Siapkan Multi-Scale Inputs
    % (Sama seperti di atas)
    [R2, ~] = downsampleAudio(Y_Real, 16000); [F2, ~] = downsampleAudio(Y_Fake, 16000);
    [~, R3] = downsampleAudio(Y_Real, 16000); [~, F3] = downsampleAudio(Y_Fake, 16000);
    R_Mel = stftMelBlock(Y_Real); F_Mel = stftMelBlock(Y_Fake);
    
    % 4. Forward dengan Feature Extraction
    
    % --- D1 (16k) ---
    [score_R1, feat_R1] = forwardWithFeatures(D1, Y_Real, 'waveform');
    [score_F1, feat_F1] = forwardWithFeatures(D1, Y_Fake, 'waveform');
    [lossAdv1, ~] = hingeGANLoss(score_R1, score_F1);
    lossFM1       = featureMatchingLoss(feat_R1, feat_F1);
    
    % --- D2 (8k) ---
    [score_R2, feat_R2] = forwardWithFeatures(D2, R2, 'waveform');
    [score_F2, feat_F2] = forwardWithFeatures(D2, F2, 'waveform');
    [lossAdv2, ~] = hingeGANLoss(score_R2, score_F2);
    lossFM2       = featureMatchingLoss(feat_R2, feat_F2);

    % --- D3 (4k) ---
    [score_R3, feat_R3] = forwardWithFeatures(D3, R3, 'waveform');
    [score_F3, feat_F3] = forwardWithFeatures(D3, F3, 'waveform');
    [lossAdv3, ~] = hingeGANLoss(score_R3, score_F3);
    lossFM3       = featureMatchingLoss(feat_R3, feat_F3);
    
    % --- D_Mel ---
    [score_RM, feat_RM] = forwardWithFeatures(DM, R_Mel, 'melspec');
    [score_FM, feat_FM] = forwardWithFeatures(DM, F_Mel, 'melspec');
    [lossAdvM, ~] = hingeGANLoss(score_RM, score_FM);
    lossFMM       = featureMatchingLoss(feat_RM, feat_FM);
    
    % 5. Total Generator Loss
    % Bobot lambda (biasanya lambda_fm = 2, lambda_adv = 1, lambda_recon = 45)
    % Sesuaikan dengan paper atau percobaan.
    
    lossAdv_Total = lossAdv1 + lossAdv2 + lossAdv3 + lossAdvM;
    lossFM_Total  = 2 * (lossFM1 + lossFM2 + lossFM3 + lossFMM); % Bobot FM biasanya 2
    
    lossG_Total = (45 * lossRecon) + lossAdv_Total + lossFM_Total;
    
    % 6. Hitung Gradien
    learnables = [netG.Learnables; netP.Learnables];
    gradsAll = dlgradient(lossG_Total, learnables);
    gradsG.G = gradsAll(1:size(netG.Learnables,1), :);
    gradsG.P = gradsAll(size(netG.Learnables,1)+1:end, :);
end