function [y_out, sr_out] = downsampleAudio(x_in, targetSR)
    % downsampleAudio: Resampling aman untuk input 3D [Time, 1, Batch]
    % Input: x_in (dlarray atau numeric)
    % Output: y_out (dlarray format 'TCB')
    
    origSR = 16000; % Default samplerate project ini
    
    % 1. Extract Data (Jika dlarray)
    if isa(x_in, 'dlarray')
        x_val = extractdata(x_in);
    else
        x_val = x_in;
    end
    
    % 2. HANDLING DIMENSI (Penyebab Error Sebelumnya)
    % Data masuk sebagai [Time, 1, Batch] (3 Dimensi)
    % Kita harus ubah jadi [Time, Batch] (2 Dimensi) agar 'resample' jalan.
    % Kita gunakan 'squeeze' untuk membuang dimensi channel 1.
    x_val_2d = squeeze(x_val); 
    
    % Jaga-jaga: Jika batch size = 1, squeeze mungkin bikin jadi vektor baris/kolom.
    % Pastikan dia kolom [Time x 1] minimal.
    if isvector(x_val_2d)
        x_val_2d = x_val_2d(:); % Paksa jadi kolom
    end

    % 3. Lakukan Resampling
    if targetSR == origSR
        y_val_2d = x_val_2d;
    else
        % resample(data, p, q) -> Resample rasional
        [p, q] = rat(targetSR / origSR);
        
        % Fungsi ini bekerja kolom-per-kolom (Batch aman)
        y_val_2d = resample(double(x_val_2d), p, q);
        
        % Kembalikan ke single precision (hemat memori GPU)
        y_val_2d = single(y_val_2d);
    end
    
    % 4. KEMBALIKAN KE FORMAT 3D [Time, 1, Batch]
    % Kita harus tahu berapa jumlah batch-nya
    [newTime, batchSize] = size(y_val_2d);
    
    % Reshape kembali: [Time, Channel=1, Batch]
    y_val_3d = reshape(y_val_2d, newTime, 1, batchSize);
    
    % Bungkus jadi dlarray TCB
    y_out = dlarray(y_val_3d, 'TCB');
    sr_out = targetSR;
end