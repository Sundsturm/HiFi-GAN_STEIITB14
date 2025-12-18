% Downsampling Frequency
function [audio_8k, audio_4k] = downsampleAudio(audio_16k, fs_input)
    % downsampleAudio Mengubah sampling rate audio dengan faktor 2 bertahap.
    %
    % Input:
    %   audio_16k : Sinyal audio asli (Vector / dlarray)
    %   fs_input  : Sampling rate asli (harus 16000 untuk skenario ini)
    %
    % Output:
    %   audio_8k  : Audio hasil downsample ke 8 kHz
    %   audio_4k  : Audio hasil downsample ke 4 kHz
    
    if nargin < 2
        fs_input = 16000;
    end
    
    % Validasi input apakah dlarray (Deep Learning Array) atau Double biasa
    isDlarray = isa(audio_16k, 'dlarray');
    if isDlarray
        % Konversi ke double dulu untuk proses signal processing MATLAB standar
        data_in = extractdata(audio_16k);
        % Pastikan bentuk kolom (Samples x 1)
        if size(data_in, 2) > size(data_in, 1)
            data_in = data_in'; 
        end
    else
        data_in = audio_16k;
    end

    % --- TAHAP 1: 16 kHz ke 8 kHz ---
    % Menggunakan 'resample'. Argumen (1, 2) berarti dikali 1 lalu dibagi 2.
    % Ini menerapkan filter anti-aliasing otomatis.
    audio_8k_raw = resample(data_in, 1, 2);
    
    % --- TAHAP 2: 8 kHz ke 4 kHz ---
    % Inputnya adalah hasil dari tahap 1
    audio_4k_raw = resample(audio_8k_raw, 1, 2);
    
    % --- Format Output ---
    % Jika input awal adalah dlarray, kembalikan sebagai dlarray
    % Biasanya format TCB (Time, Channel, Batch) atau CB (Channel, Batch)
    % Di sini kita kembalikan sebagai vektor kolom standar
    if isDlarray
        audio_8k = dlarray(single(audio_8k_raw));
        audio_4k = dlarray(single(audio_4k_raw));
    else
        audio_8k = audio_8k_raw;
        audio_4k = audio_4k_raw;
    end
    
    % Tampilkan info (Opsional)
    fprintf('Input: %d samples (16 kHz)\n', length(data_in));
    fprintf('Output 1: %d samples (8 kHz)\n', length(audio_8k_raw));
    fprintf('Output 2: %d samples (4 kHz)\n', length(audio_4k_raw));
end