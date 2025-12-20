function [lossG, lossD] = hingeGANLoss(scoreReal, scoreFake)
    % hingeGANLoss: Menghitung Adversarial Loss (Hinge Version)
    % Robust Version: Menggunakan max(0, x) alih-alih relu()
    
    % 1. FORCE REAL
    % Mencegah error jika ada sisa-sisa bilangan kompleks dari konvolusi
    scoreReal = real(scoreReal);
    scoreFake = real(scoreFake);
    
    % 2. DISCRIMINATOR LOSS
    % Formula: E[ReLU(1 - D(x))] + E[ReLU(1 + D(G(z)))]
    % Kita gunakan max(0, ...) sebagai pengganti relu(...) agar lebih aman
    
    lossD_Real = mean(max(0, 1 - scoreReal), 'all');
    lossD_Fake = mean(max(0, 1 + scoreFake), 'all');
    
    lossD = lossD_Real + lossD_Fake;
    
    % 3. GENERATOR LOSS
    % Formula: -E[D(G(z))]
    lossG = -mean(scoreFake, 'all');
end