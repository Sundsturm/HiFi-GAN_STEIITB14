function lossFM = featureMatchingLoss(featsReal, featsFake)
    % featureMatchingLoss Menghitung L1 distance antar layer internal
    % Input berupa Cell Array berisi dlarray dari tiap layer.
    
    lossFM = 0;
    numLayers = numel(featsReal);
    
    for i = 1:numLayers
        F_Real = featsReal{i};
        F_Fake = featsFake{i};
        
        % Eq (3): Mean Absolute Error (L1)
        % Normalisasi 1/Ni sudah otomatis dilakukan oleh 'mean'.
        lossFM = lossFM + mean(abs(F_Real - F_Fake), 'all');
    end
    
    % Opsional: Bagi dengan jumlah layer agar skalanya tidak meledak
    % lossFM = lossFM / numLayers; 
end