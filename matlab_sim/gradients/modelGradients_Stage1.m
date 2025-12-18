function [grads, totalLoss] = modelGradients_Stage1(netG, X, Y)
    Y_Pred = forward(netG, X);
    
    % Loss: L1 + Spectrogram
    lossL1 = l1SampleLoss(Y, Y_Pred);
    lossSpec = dualScaleSpectrogramLoss(Y, Y_Pred);
    
    totalLoss = lossL1 + lossSpec;
    grads = dlgradient(totalLoss, netG.Learnables);
end