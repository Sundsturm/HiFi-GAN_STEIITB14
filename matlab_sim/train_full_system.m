%% INCLUDE FOLDERS
addpath('utils\')
addpath('models\')
addpath('gradients\')
%% KONFIGURASI UMUM & HARDWARE
clear; rehash;

% DUMMY DATASET
pathToNoisy = "./data/dummy/Noisy";
pathToClean = "./data/dummy/Clean";

% Parameter Hardware (RTX 3050 Laptop)
miniBatchSize = 2;         % Turunkan ke 2 jika Out of Memory
segmentLength = 8192;      % 0.5 detik di 16kHz (Hemat VRAM)
executionEnvironment = "gpu";

% Datastore preparation
disp("Menyiapkan datastore....");
adsNoisy = audioDatastore(pathToNoisy, 'IncludeSubfolders', true, 'FileExtensions', '.wav');
adsClean = audioDatastore(pathToClean, 'IncludeSubfolders', true, 'FileExtensions', '.wav');
if isempty(adsNoisy.Files)
    error('Tidak ada file .wav ditemukan di %s', pathToNoisy);
end
dsCombined = combine(adsNoisy, adsClean);
ds = transform(dsCombined, @(data) preprocessData(data{1}, data{2}, segmentLength));

% Setup MinibatchQueue (Crop otomatis ke 8192 sample)
mbq = minibatchqueue(ds, ...
    'MiniBatchSize', miniBatchSize, ...
    'MiniBatchFormat', 'TCB', ...
    'PartialMiniBatch', 'discard', ...
    'DispatchInBackground', true, ...
    'OutputEnvironment', executionEnvironment, ...
    'OutputCast', 'single'); % Penting: Single Precision
disp("Datastore siap!");

%% --- STAGE 1: PRE-TRAIN GENERATOR (WaveNet Only) ---
% --- STAGE 1: PRE-TRAIN GENERATOR (WaveNet Only) ---
disp('=== MEMULAI STAGE 1: Pre-training Generator ===');

lgraphG = createFinalWaveNetGenerator(); 
netG = dlnetwork(lgraphG);

learnRate = 0.001; 
avgG = []; avgSqG = [];
numStepsStage1 = 100; % Atau kurangi untuk tes

start = tic;

% --- CORE LOOP STAGE 1 (DENGAN FIX DIMENSI) ---
for step = 1:numStepsStage1
    if ~hasdata(mbq), reset(mbq); end
    
    % 1. Ambil data dari queue
    [dlX_Noisy, dlY_Clean] = next(mbq);
    
    % === BAGIAN PENTING: FIX DIMENSION ===
    % Memaksa dimensi menjadi [8192, 1, 2]
    % Tanpa ini, dimensi adalah [8192, 2], dan Network akan error.
    
    % A. Strip metadata dlarray
    xRaw = extractdata(dlX_Noisy);
    yRaw = extractdata(dlY_Clean);
    
    % B. Reshape Paksa: [Time, Channel=1, Batch]
    % segmentLength harus sesuai setting (8192)
    xRaw = reshape(xRaw, segmentLength, 1, []); 
    yRaw = reshape(yRaw, segmentLength, 1, []);
    
    % C. Bungkus ulang jadi dlarray dengan format TCB
    dlX_Noisy = dlarray(xRaw, 'TCB');
    dlY_Clean = dlarray(yRaw, 'TCB');
    % === SELESAI FIX ===
    
    % Data Augmentation
    scale = 0.9 + (0.2 * rand);
    dlX_Noisy = dlX_Noisy * scale;
    dlY_Clean = dlY_Clean * scale;
    
    % Evaluasi Gradient
    [grads, loss] = dlfeval(@modelGradients_Stage1, netG, dlX_Noisy, dlY_Clean);
    
    % Update
    [netG, avgG, avgSqG] = adamupdate(netG, grads, avgG, avgSqG, step, learnRate);
    
    % Monitoring
    if mod(step, 100) == 0
        D = duration(0, 0, toc(start), 'Format', 'hh:mm:ss');
        disp(['Stage 1 | Step: ' num2str(step) ' | Loss: ' num2str(extractdata(loss)) ' | Time: ' char(D)]);
    end
    
    % Save Checkpoint
    if mod(step, 10000) == 0
        save('checkpoint_stage1.mat', 'netG', 'step');
    end
end
disp('Stage 1 Selesai.');
save('final_stage1.mat', 'netG');

%% --- STAGE 2: JOINT TRAINING (Generator + PostNet) ---
disp('=== MEMULAI STAGE 2: Joint Training (G + PostNet) ===');

% Load Model Stage 1
load('final_stage1.mat', 'netG'); 

% Inisialisasi PostNet
lgraphP = createSpecificPostNet(1); 
netP = dlnetwork(lgraphP);

% Optimizer
learnRate = 0.0001;
avgG = []; avgSqG = [];
avgP = []; avgSqP = [];
numStepsStage2 = 100; % Sesuaikan kebutuhan

start = tic;

% --- CORE LOOP STAGE 2 ---
for step = 1:numStepsStage2
    if ~hasdata(mbq), reset(mbq); end
    
    % 1. Ambil Data
    [dlX_Noisy, dlY_Clean] = next(mbq);
    
    % === FIX DIMENSION (WAJIB ADA DI STAGE 2 JUGA) ===
    % Mengubah [Time, Batch] -> [Time, Channel=1, Batch]
    xRaw = extractdata(dlX_Noisy);
    yRaw = extractdata(dlY_Clean);
    
    xRaw = reshape(xRaw, segmentLength, 1, []); 
    yRaw = reshape(yRaw, segmentLength, 1, []);
    
    dlX_Noisy = dlarray(xRaw, 'TCB');
    dlY_Clean = dlarray(yRaw, 'TCB');
    % === SELESAI FIX ===
    
    % Augmentasi
    scale = 0.9 + (0.2 * rand);
    dlX_Noisy = dlX_Noisy * scale;
    dlY_Clean = dlY_Clean * scale;
    
    % 2. Evaluasi Gradient
    [grads, lossTotal] = dlfeval(@modelGradients_Stage2, netG, netP, dlX_Noisy, dlY_Clean);
    
    % 3. Update
    [netG, avgG, avgSqG] = adamupdate(netG, grads.G, avgG, avgSqG, step, learnRate);
    [netP, avgP, avgSqP] = adamupdate(netP, grads.P, avgP, avgSqP, step, learnRate);
    
    % Monitoring
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
disp('=== MEMULAI STAGE 3: Adversarial Training ===');

% 1. Load Model Terakhir
load('final_stage2.mat', 'netG', 'netP');

% 2. Inisialisasi Discriminators
% Pastikan file createWaveformDiscriminator sudah yang versi baru (tanpa NumGroups)
lgraphD_W = createWaveformDiscriminator();
netD_W1 = dlnetwork(lgraphD_W); 
netD_W2 = dlnetwork(lgraphD_W); 
netD_W3 = dlnetwork(lgraphD_W);

% Pastikan file createMelSpecDiscriminator sudah yang versi baru (tanpa NaN)
lgraphD_Mel = createMelSpecDiscriminator(); 
netD_Mel = dlnetwork(lgraphD_Mel);

% 3. Optimizer Settings
lr_G = 0.00001; % Generator LR Kecil
lr_D = 0.001;   % Discriminator LR Besar

% State optimizers
avgG=[]; avgSqG=[]; avgP=[]; avgSqP=[];
avgD1=[]; avgSqD1=[]; avgD2=[]; avgSqD2=[]; avgD3=[]; avgSqD3=[]; avgDM=[]; avgSqDM=[];

numStepsStage3 = 100; % 50K steps
start = tic;

% --- CORE LOOP STAGE 3 ---
for step = 1:numStepsStage3
    
    % === A. UPDATE DISCRIMINATOR (2 KALI) ===
    for k = 1:2
        if ~hasdata(mbq), reset(mbq); end
        [dlX_Noisy, dlY_Clean] = next(mbq);
        
        % --- FIX DIMENSION (WAJIB ADA) ---
        % Mengubah [Time, Batch] (8192x2) -> [Time, Channel=1, Batch] (8192x1x2)
        xRaw = extractdata(dlX_Noisy);
        yRaw = extractdata(dlY_Clean);
        
        xRaw = reshape(xRaw, segmentLength, 1, []); 
        yRaw = reshape(yRaw, segmentLength, 1, []);
        
        dlX_Noisy = dlarray(xRaw, 'TCB');
        dlY_Clean = dlarray(yRaw, 'TCB');
        % ---------------------------------
        
        % Hitung Gradient Discriminator
        % (Menggunakan nama file sesuai error log Anda: gradients_DiscriminatorOnlyStage3)
        [gradsD, lossD_val] = dlfeval(@gradients_DiscriminatorOnlyStage3, ...
            netG, netP, netD_W1, netD_W2, netD_W3, netD_Mel, ...
            dlX_Noisy, dlY_Clean);
            
        % Update 4 Discriminator
        [netD_W1, avgD1, avgSqD1] = adamupdate(netD_W1, gradsD.D1, avgD1, avgSqD1, step, lr_D);
        [netD_W2, avgD2, avgSqD2] = adamupdate(netD_W2, gradsD.D2, avgD2, avgSqD2, step, lr_D);
        [netD_W3, avgD3, avgSqD3] = adamupdate(netD_W3, gradsD.D3, avgD3, avgSqD3, step, lr_D);
        [netD_Mel, avgDM, avgSqDM] = adamupdate(netD_Mel, gradsD.DM, avgDM, avgSqDM, step, lr_D);
    end
    
    % === B. UPDATE GENERATOR (1 KALI) ===
    if ~hasdata(mbq), reset(mbq); end
    [dlX_Noisy, dlY_Clean] = next(mbq);
    
    % --- FIX DIMENSION (WAJIB ADA JUGA DI SINI) ---
    xRaw = extractdata(dlX_Noisy);
    yRaw = extractdata(dlY_Clean);
    xRaw = reshape(xRaw, segmentLength, 1, []); 
    yRaw = reshape(yRaw, segmentLength, 1, []);
    dlX_Noisy = dlarray(xRaw, 'TCB');
    dlY_Clean = dlarray(yRaw, 'TCB');
    % ----------------------------------------------
    
    % Hitung Gradient Generator
    [gradsG, lossG_val] = dlfeval(@gradients_GeneratorOnlyStage3, ...
            netG, netP, netD_W1, netD_W2, netD_W3, netD_Mel, ...
            dlX_Noisy, dlY_Clean);
            
    % Update G dan P
    [netG, avgG, avgSqG] = adamupdate(netG, gradsG.G, avgG, avgSqG, step, lr_G);
    [netP, avgP, avgSqP] = adamupdate(netP, gradsG.P, avgP, avgSqP, step, lr_G);
    
    % Monitoring
    if mod(step, 50) == 0
        D = duration(0, 0, toc(start), 'Format', 'hh:mm:ss');
        disp(['Stage 3 | Step: ' num2str(step) ' | Loss G: ' num2str(extractdata(lossG_val)) ' | Loss D: ' num2str(extractdata(lossD_val)) ' | Time: ' char(D)]);
    end
    
    if mod(step, 5000) == 0
        save('checkpoint_stage3.mat', 'netG', 'netP', 'netD_W1', 'step');
    end
end
save('final_model_complete.mat', 'netG', 'netP');
disp('PELATIHAN SELESAI!');

function dataOut = preprocessData(xIn, yIn, targetLen)
    % preprocessData: Force Mono & Bungkus ke Table
    
    % 1. PAKSA MONO (Ambil kolom 1 saja, abaikan kolom 2 jika stereo)
    % Ini lebih aman daripada 'mean'
    xIn = xIn(:, 1);
    yIn = yIn(:, 1);
    
    % 2. Random Crop
    numSamples = size(xIn, 1);
    if numSamples >= targetLen
        startIdx = randi(numSamples - targetLen + 1);
        range = startIdx : (startIdx + targetLen - 1);
        xOut = xIn(range, :);
        yOut = yIn(range, :);
    else
        % Zero Padding
        padAmt = targetLen - numSamples;
        xOut = [xIn; zeros(padAmt, 1)];
        yOut = [yIn; zeros(padAmt, 1)];
    end
    
    % 3. CEK DIMENSI (Debug)
    % Jika masih error, teks ini akan muncul di Command Window dan memberi tahu kita ukurannya
    if size(xOut, 2) ~= 1
        warning('Data masih bukan Mono! Size: %d x %d', size(xOut, 1), size(xOut, 2));
    end
    
    % 4. RETURN TABLE
    dataOut = table({xOut}, {yOut}, 'VariableNames', {'Input', 'Target'});
end