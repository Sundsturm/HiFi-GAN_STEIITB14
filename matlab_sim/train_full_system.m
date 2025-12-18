%% INCLUDE FOLDERS
addpath("utils\")
addpath("models\")
addpath("gradients\")

%% KONFIGURASI UMUM & HARDWARE
clear; clc;

% Parameter Hardware (RTX 3050 Laptop)
miniBatchSize = 2;         % Turunkan ke 2 jika Out of Memory
segmentLength = 8192;      % 0.5 detik di 16kHz (Hemat VRAM)
executionEnvironment = "gpu";

% Load Dataset (Asumsi 'ds' sudah siap dari folder audio)
% ds = audioDatastore('path_to_wavs', 'FileExtensions', '.wav');
% Untuk demo, kita skip definisi ds. Pastikan ds sudah ada di workspace.

% Setup MinibatchQueue (Crop otomatis ke 8192 sample)
mbq = minibatchqueue(ds, ...
    'MiniBatchSize', miniBatchSize, ...
    'MiniBatchFormat', 'TCB', ...
    'PartialMiniBatch', 'discard', ...
    'DispatchInBackground', true, ...
    'OutputEnvironment', executionEnvironment, ...
    'OutputCast', 'single'); % Penting: Single Precision

%% --- STAGE 1: PRE-TRAIN GENERATOR (WaveNet Only) ---
% "First, we train the feed-forward WaveNet for 500K steps with LR 0.001"
% "Using L1 loss and spectrogram loss."

disp('=== MEMULAI STAGE 1: Pre-training Generator ===');

% 1. Inisialisasi
lgraphG = createFinalWaveNetGenerator(); 
netG = dlnetwork(lgraphG);

% Optimizer
learnRate = 0.001; 
avgG = []; avgSqG = [];
numStepsStage1 = 500000; % Bisa dikurangi, misal 20000 untuk tes

% Loop
start = tic;
for step = 1:numStepsStage1
    % Reset epoch jika data habis (Infinite Loop logic)
    if ~hasdata(mbq), reset(mbq); end
    
    [dlX_Noisy, dlY_Clean] = next(mbq);
    
    % Data Augmentation (Simple Amplitude Scaling)
    % "Data augmentation" disebut di stage 2, tapi bagus juga di sini
    scale = 0.9 + (0.2 * rand); % Random 0.9 s/d 1.1
    dlX_Noisy = dlX_Noisy * scale;
    dlY_Clean = dlY_Clean * scale;
    
    % Evaluasi Gradient (Hanya Generator)
    [grads, loss] = dlfeval(@modelGradients_Stage1, netG, dlX_Noisy, dlY_Clean);
    
    % Update
    [netG, avgG, avgSqG] = adamupdate(netG, grads, avgG, avgSqG, step, learnRate);
    
    % Monitoring
    if mod(step, 100) == 0
        D = duration(0, 0, toc(start), 'Format', 'hh:mm:ss');
        disp(['Stage 1 | Step: ' num2str(step) ' | Loss: ' num2str(extractdata(loss)) ' | Time: ' char(D)]);
    end
    
    % Save Checkpoint tiap 10.000 steps
    if mod(step, 10000) == 0
        save('checkpoint_stage1.mat', 'netG', 'step');
    end
end
disp('Stage 1 Selesai. Model disimpan.');
save('final_stage1.mat', 'netG');

%% --- STAGE 2: JOINT TRAINING (Generator + PostNet) ---
% "Then we train together with the postnet... for 500K steps, with LR 0.0001"

disp('=== MEMULAI STAGE 2: Joint Training (G + PostNet) ===');

% 1. Load Model Stage 1
load('final_stage1.mat', 'netG'); % Load netG yang sudah pintar

% 2. Inisialisasi PostNet (Baru)
lgraphP = createSpecificPostNet(128); % Asumsi output G adalah 128 channel
netP = dlnetwork(lgraphP);

% Optimizer (Reset state, LR baru)
learnRate = 0.0001;
avgG = []; avgSqG = [];
avgP = []; avgSqP = [];
numStepsStage2 = 500000;

start = tic;
for step = 1:numStepsStage2
    if ~hasdata(mbq), reset(mbq); end
    [dlX_Noisy, dlY_Clean] = next(mbq);
    
    % Augmentasi
    dlX_Noisy = dlX_Noisy * (0.9 + 0.2*rand);
    dlY_Clean = dlY_Clean * (0.9 + 0.2*rand);
    
    % Evaluasi Gradient (G + PostNet)
    [grads, lossTotal] = dlfeval(@modelGradients_Stage2, netG, netP, dlX_Noisy, dlY_Clean);
    
    % Update G dan P bersamaan
    [netG, avgG, avgSqG] = adamupdate(netG, grads.G, avgG, avgSqG, step, learnRate);
    [netP, avgP, avgSqP] = adamupdate(netP, grads.P, avgP, avgSqP, step, learnRate);
    
    if mod(step, 100) == 0
        D = duration(0, 0, toc(start), 'Format', 'hh:mm:ss');
        disp(['Stage 2 | Step: ' num2str(step) ' | Loss: ' num2str(extractdata(lossTotal)) ' | Time: ' char(D)]);
    end
    
    if mod(step, 10000) == 0
        save('checkpoint_stage2.mat', 'netG', 'netP', 'step');
    end
end
save('final_stage2.mat', 'netG', 'netP');

%% --- STAGE 3: ADVERSARIAL TRAINING (Full System) ---
% "Finally, train generator at LR 0.00001 with four discriminators at LR 0.001"
% "Update discriminators twice for every step of the generator"

disp('=== MEMULAI STAGE 3: Adversarial Training ===');

% 1. Load G dan P dari Stage 2
load('final_stage2.mat', 'netG', 'netP');

% 2. Inisialisasi Discriminators
lgraphD_W = createWaveformDiscriminator();
netD_W1 = dlnetwork(lgraphD_W); 
netD_W2 = dlnetwork(lgraphD_W); 
netD_W3 = dlnetwork(lgraphD_W);
lgraphD_Mel = createMelSpecDiscriminator(); 
netD_Mel = dlnetwork(lgraphD_Mel);

% 3. Optimizer Settings
lr_G = 0.00001; % Generator LR Kecil
lr_D = 0.001;   % Discriminator LR Besar

% State optimizers
avgG=[]; avgSqG=[]; avgP=[]; avgSqP=[];
avgD1=[]; avgSqD1=[]; avgD2=[]; avgSqD2=[]; avgD3=[]; avgSqD3=[]; avgDM=[]; avgSqDM=[];

numStepsStage3 = 50000; % 50K steps

start = tic;
for step = 1:numStepsStage3
    
    % === A. UPDATE DISCRIMINATOR (2 KALI) ===
    % "Update discriminators twice for every step of the generator"
    for k = 1:2
        if ~hasdata(mbq), reset(mbq); end
        [dlX_Noisy, dlY_Clean] = next(mbq);
        
        % Hitung Gradient Discriminator Saja
        [gradsD, lossD_val] = dlfeval(@gradients_DiscriminatorOnly, ...
            netG, netP, netD_W1, netD_W2, netD_W3, netD_Mel, ...
            dlX_Noisy, dlY_Clean);
            
        % Update 4 Discriminator
        [netD_W1, avgD1, avgSqD1] = adamupdate(netD_W1, gradsD.D1, avgD1, avgSqD1, step, lr_D);
        [netD_W2, avgD2, avgSqD2] = adamupdate(netD_W2, gradsD.D2, avgD2, avgSqD2, step, lr_D);
        [netD_W3, avgD3, avgSqD3] = adamupdate(netD_W3, gradsD.D3, avgD3, avgSqD3, step, lr_D);
        [netD_Mel, avgDM, avgSqDM] = adamupdate(netD_Mel, gradsD.DM, avgDM, avgSqDM, step, lr_D);
    end
    
    % === B. UPDATE GENERATOR (1 KALI) ===
    % Gunakan data batch baru atau pakai yang terakhir (di sini pakai batch baru biar fresh)
    if ~hasdata(mbq), reset(mbq); end
    [dlX_Noisy, dlY_Clean] = next(mbq);
    
    % Hitung Gradient Generator Saja
    [gradsG, lossG_val] = dlfeval(@gradients_GeneratorOnly, ...
            netG, netP, netD_W1, netD_W2, netD_W3, netD_Mel, ...
            dlX_Noisy, dlY_Clean);
            
    % Update G dan P
    [netG, avgG, avgSqG] = adamupdate(netG, gradsG.G, avgG, avgSqG, step, lr_G);
    [netP, avgP, avgSqP] = adamupdate(netP, gradsG.P, avgP, avgSqP, step, lr_G);
    
    % Monitoring
    if mod(step, 50) == 0
        D = duration(0, 0, toc(start), 'Format', 'hh:mm:ss');
        disp(['Stage 3 | Step: ' num2str(step) ' | Loss G: ' num2str(extractdata(lossG_val)) ' | Loss D: ' num2str(extractdata(lossD_val))]);
    end
    
    if mod(step, 5000) == 0
        save('checkpoint_stage3.mat', 'netG', 'netP', 'netD_W1', 'step');
    end
end
save('final_model_complete.mat', 'netG', 'netP');
disp('PELATIHAN SELESAI!');