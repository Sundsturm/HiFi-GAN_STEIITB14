function [lossG, lossD] = hingeGANLoss(scoreReal, scoreFake)
    % hingeGANLoss Menghitung Hinge Loss sesuai persamaan (1) dan (2)
    
    % --- DISCRIMINATOR LOSS ---
    % Eq (2): L_D = max(0, 1 + D(Fake)) + max(0, 1 - D(Real))
    % Artinya: Real harus > 1, Fake harus < -1.
    lossD_Real = mean(relu(1 - scoreReal), 'all');
    lossD_Fake = mean(relu(1 + scoreFake), 'all');
    lossD = lossD_Real + lossD_Fake;
    
    % --- GENERATOR LOSS ---
    % Eq (1): L_G = max(0, 1 - D(Fake))
    % Artinya: Generator ingin menipu D agar menilai Fake > 1.
    % Jika D(Fake) > 1, loss = 0.
    lossG = mean(relu(1 - scoreFake), 'all');
end