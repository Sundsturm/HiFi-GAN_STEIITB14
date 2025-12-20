function [score, features] = forwardWithFeatures(net, X, type)
    % forwardWithFeatures: Versi FIXED OUTPUT COUNT
    % Memastikan jumlah request layer == jumlah variabel output
    
    if strcmp(type, 'waveform')
        % --- WAVEFORM DISCRIMINATOR ---
        % Kita butuh 7 Output total:
        % 1. Score (dari 'global_pool')
        % 2-7. Fitur Internal (6 layer)
        
        layerNames = {
            'global_pool', ...  % Output 1 (Score)
            'lrelu_1', ...      % Output 2
            'lrelu_2', ...      % Output 3
            'lrelu_3', ...      % Output 4
            'lrelu_4', ...      % Output 5
            'lrelu_5', ...      % Output 6
            'final_conv'        % Output 7
        };
        
        % Panggil forward dengan 7 variabel penampung
        [score, f1, f2, f3, f4, f5, f6] = forward(net, X, 'Outputs', layerNames);
        
        % Bungkus fitur (f1 s/d f6)
        features = {f1, f2, f3, f4, f5, f6};
        
    elseif strcmp(type, 'melspec')
        % --- MEL-SPEC DISCRIMINATOR ---
        % Kita butuh 5 Output total.
        % Score diambil dari layer terakhir ('final_conv').
        % Fitur adalah semua 5 layer tersebut.
        
        layerNames = {
            'stack1_mult', ...
            'stack2_mult', ...
            'stack3_mult', ...
            'stack4_mult', ...
            'final_conv'   % Ini sekaligus Score
        };
        
        % Panggil forward dengan 5 variabel penampung
        [f1, f2, f3, f4, f5] = forward(net, X, 'Outputs', layerNames);
        
        % Score adalah output terakhir (f5)
        score = f5;
        
        % Fitur adalah kelimanya
        features = {f1, f2, f3, f4, f5};
        
    else
        error('Type harus "waveform" atau "melspec"');
    end
end