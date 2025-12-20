function dlMelLog = stftMelBlock(dlWaveform)
    % stftMelBlock: Robust Version dengan AUTO-RESHAPE
    % Input: dlWaveform [Time, 1, Batch]
    % Output: dlMelLog [NumMels, Time, 1, Batch] (SSCB)
    
    % 1. Persiapan Data
    dlWaveform = real(dlWaveform);
    
    % Parameter (Sesuai HiFi-GAN)
    n_fft = 1024;
    hop_length = 256;
    win_length = 1024;
    n_mels = 80;
    fs = 16000;
    
    window = hann(win_length, 'periodic');
    
    % 2. HITUNG STFT
    % Output dlstft bisa [Freq, Time, Batch] atau [Freq, Time, 1, Batch]
    % tergantung apakah MATLAB melakukan squeeze atau tidak.
    dlSpecComplex = dlstft(dlWaveform, ...
        'Window', window, ...
        'OverlapLength', win_length - hop_length, ...
        'FFTLength', n_fft);
        
    dlMag = abs(dlSpecComplex);
    
    % 3. SIAPKAN MEL FILTERBANK (Manual)
    persistent melWeights
    if isempty(melWeights)
        weightsRaw = createMelFilterBank(fs, n_fft, n_mels);
        melWeights = dlarray(single(weightsRaw)); 
    end
    
    % 4. MATRIKS MULTIPLICATION (FIXED)
    
    % Ambil ukuran Freq (513) dan Time (misal 32) dari output STFT asli
    % Kita gunakan size(..., 2) untuk memastikan Time diambil dengan benar
    nFreq = size(dlMag, 1);
    nTime = size(dlMag, 2); 
    
    % Pipihkan dimensi belakang (Batch/Channel) menjadi satu antrian panjang
    % [513, Time * Batch]
    dlMagReshaped = reshape(dlMag, nFreq, []); 
    
    % Kalikan: [80 x 513] * [513 x N] = [80 x N]
    dlMelRaw = melWeights * dlMagReshaped;
    
    % --- SOLUSI ERROR "NUMBER OF ELEMENTS" ---
    % Jangan gunakan variabel 'batchSize' manual. 
    % Gunakan [] pada posisi terakhir. 
    % MATLAB akan otomatis menghitung batch size berdasarkan sisa elemen.
    % Target: [Mel, Time, 1, Batch]
    
    dlMel = reshape(dlMelRaw, n_mels, nTime, 1, []);
    
    % 5. Log Scale & Output
    epsilon = 1e-5;
    dlMelLog = log(dlMel + epsilon);
    
    % Labeli SSCB (Spatial, Spatial, Channel, Batch)
    dlMelLog = dlarray(extractdata(dlMelLog), 'SSCB');
end

%% --- LOCAL FUNCTION: MANUAL MEL FILTERBANK ---
function weights = createMelFilterBank(fs, n_fft, n_mels)
    f_min = 0;
    f_max = fs / 2;
    mel_min = 2595 * log10(1 + f_min / 700);
    mel_max = 2595 * log10(1 + f_max / 700);
    mel_points = linspace(mel_min, mel_max, n_mels + 2);
    hz_points = 700 * (10.^(mel_points / 2595) - 1);
    bin_points = floor((n_fft + 1) * hz_points / fs);
    
    num_bins = floor(n_fft / 2) + 1;
    weights = zeros(n_mels, num_bins);
    
    for i = 1:n_mels
        b_left = bin_points(i);
        b_center = bin_points(i+1);
        b_right = bin_points(i+2);
        
        for k = b_left:b_center
            if (k >= 0) && (k < num_bins)
                weights(i, k+1) = (k - b_left) / (b_center - b_left);
            end
        end
        for k = b_center:b_right
            if (k >= 0) && (k < num_bins)
                weights(i, k+1) = (b_right - k) / (b_right - b_center);
            end
        end
    end
end