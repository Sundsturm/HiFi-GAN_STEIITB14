% L1 Sample Loss
function loss = l1SampleLoss(YTrue, YPred)
    % YTrue: Audio Asli (dlarray: Time x 1 x Batch)
    % YPred: Audio Generator (dlarray: Time x 1 x Batch)
    
    % Hitung Mean Absolute Error
    loss = mean(abs(YTrue - YPred), 'all');
end