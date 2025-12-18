function [grads, totalLoss] = modelGradients_Stage2(netG, netP, X, Y)
    Y_Mid = forward(netG, X);
    Y_Final = forward(netP, Y_Mid);
    
    % Loss dihitung pada kedua output (Intermediate & Final) untuk stabilitas
    lossL1_Mid = l1SampleLoss(Y, Y_Mid);
    lossSpec_Mid = dualScaleSpectrogramLoss(Y, Y_Mid);
    
    lossL1_Final = l1SampleLoss(Y, Y_Final);
    lossSpec_Final = dualScaleSpectrogramLoss(Y, Y_Final);
    
    totalLoss = lossL1_Mid + lossSpec_Mid + lossL1_Final + lossSpec_Final;
    
    % Hitung gradient gabungan
    learnables = [netG.Learnables; netP.Learnables];
    gradsAll = dlgradient(totalLoss, learnables);
    
    % Pisahkan
    grads.G = gradsAll(1:size(netG.Learnables,1), :);
    grads.P = gradsAll(size(netG.Learnables,1)+1:end, :);
end