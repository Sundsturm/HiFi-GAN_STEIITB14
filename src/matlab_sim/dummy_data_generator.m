%% SETUP DUMMY DATASET GENERATOR
clear; clc;

% --- KONFIGURASI FOLDER TUJUAN ---
% Menggunakan path absolut sesuai permintaan Anda
baseDir = './data/dummy'; 

% Konfigurasi Audio
fs = 16000;          % Sampling rate
duration = 1;        % Durasi 1 detik per file
numFiles = 50;       % Jumlah file sampel

% Buat struktur folder Clean & Noisy di dalam /data/dummy
cleanDir = fullfile(baseDir, 'Clean');
noisyDir = fullfile(baseDir, 'Noisy');

% Cek dan buat folder jika belum ada struktur sub-foldernya
if ~exist(baseDir, 'dir')
    error('Folder /data/dummy tidak ditemukan! Harap buat folder tersebut terlebih dahulu.');
end
if ~exist(cleanDir, 'dir'), mkdir(cleanDir); end
if ~exist(noisyDir, 'dir'), mkdir(noisyDir); end

disp(['Sedang membuat dummy dataset di: ' baseDir]);

%% GENERATE FILES
t = (0:1/fs:duration-1/fs)';

for i = 1:numFiles
    % 1. Buat Sinyal Clean (Harmonik Sinus)
    freq1 = randi([200, 500]);
    freq2 = randi([800, 1200]);
    
    cleanAudio = 0.5 * sin(2*pi*freq1*t) + 0.3 * sin(2*pi*freq2*t);
    
    % Amplop agar halus
    envelope = hamming(length(cleanAudio));
    cleanAudio = cleanAudio .* envelope;
    
    % 2. Buat Sinyal Noisy (Clean + White Noise)
    noiseLevel = 0.2;
    noise = noiseLevel * randn(size(cleanAudio));
    noisyAudio = cleanAudio + noise;
    
    % 3. Normalisasi & Simpan
    cleanAudio = cleanAudio / max(abs(cleanAudio));
    noisyAudio = noisyAudio / max(abs(noisyAudio));
    
    filename = sprintf('sample_%03d.wav', i);
    
    audiowrite(fullfile(cleanDir, filename), cleanAudio, fs);
    audiowrite(fullfile(noisyDir, filename), noisyAudio, fs);
    
    if mod(i, 10) == 0
        fprintf('Berhasil membuat %d/%d file...\n', i, numFiles);
    end
end

disp('--- SELESAI ---');
disp('Data tersimpan di:');
disp(['Clean: ' cleanDir]);
disp(['Noisy: ' noisyDir]);