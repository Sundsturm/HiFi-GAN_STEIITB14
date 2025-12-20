function loss = featureMatchingLoss(featReal, featFake)
    % featureMatchingLoss: L1 Distance antar fitur internal
    
    loss = 0;
    
    % Loop melalui setiap layer
    for i = 1:numel(featReal)
        F_Real = featReal{i};
        F_Fake = featFake{i};
        
        % --- DEBUG CHECK ---
        % Jika error 'Invalid data type' muncul lagi, blok ini akan memberi tahu alasannya.
        if ~isa(F_Real, 'dlarray') && ~isnumeric(F_Real)
            disp(['[DEBUG ERROR] Data pada index ', num2str(i), ' bukan dlarray!']);
            disp(['Tipe data yang diterima: ', class(F_Real)]);
            error('featureMatchingLoss menerima data sampah (bukan dlarray). Cek forwardWithFeatures.');
        end
        % -------------------
        
        % Force Real (Aman karena sudah dipastikan dlarray/numeric)
        F_Real = real(F_Real);
        F_Fake = real(F_Fake);
        
        % Hitung L1 Loss
        currentLoss = mean(abs(F_Real - F_Fake), 'all');
        
        loss = loss + currentLoss;
    end
end