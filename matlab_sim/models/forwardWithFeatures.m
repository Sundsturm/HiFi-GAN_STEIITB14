function [score, features] = forwardWithFeatures(net, X, type)
    % Wrapper untuk mengambil output akhir (Score) DAN fitur internal
    
    if strcmp(type, 'waveform')
        % Nama layer internal dari fungsi createWaveformDiscriminator
        layerNames = {'lrelu_1', 'lrelu_2', 'lrelu_3', 'lrelu_4', 'lrelu_5', 'lrelu_6', 'conv_7'};
    else % 'melspec'
        % Nama layer internal dari fungsi createMelSpecDiscriminator
        layerNames = {'stack1_mult', 'stack2_mult', 'stack3_mult', 'stack4_mult', 'final_conv'};
    end
    
    % Panggil forward dengan multiple outputs
    % Syntaks ini mengambil output dari layerNames DAN output standard (Score)
    outputs = cell(1, length(layerNames));
    [score, outputs{:}] = forward(net, X, 'Outputs', layerNames);
    
    features = outputs;
end