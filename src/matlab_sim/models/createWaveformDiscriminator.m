function lgraph = createWaveformDiscriminator()
    % Waveform Discriminator dengan GLOBAL AVERAGE POOLING
    % Output: Skalar tunggal (1x1) per batch, bukan sequence.
    
    lgraph = layerGraph();
    
    % 1. Input Layer
    lgraph = addLayers(lgraph, sequenceInputLayer(1, 'Name', 'input', 'Normalization', 'none'));
    
    lastLayerName = 'input';
    
    % --- DOWN-SAMPLING BLOCKS ---
    % Sama seperti sebelumnya (Conv -> LeakyReLU)
    
    % Layer 1: Downsample 4x
    lgraph = addLayers(lgraph, convolution1dLayer(15, 64, 'Stride', 4, 'Padding', 'same', 'Name', 'conv_1'));
    lgraph = addLayers(lgraph, leakyReluLayer(0.2, 'Name', 'lrelu_1'));
    lgraph = connectLayers(lgraph, lastLayerName, 'conv_1');
    lgraph = connectLayers(lgraph, 'conv_1', 'lrelu_1');
    lastLayerName = 'lrelu_1';
    
    % Layer 2: Downsample 4x
    lgraph = addLayers(lgraph, convolution1dLayer(41, 128, 'Stride', 4, 'Padding', 'same', 'Name', 'conv_2'));
    lgraph = addLayers(lgraph, leakyReluLayer(0.2, 'Name', 'lrelu_2'));
    lgraph = connectLayers(lgraph, lastLayerName, 'conv_2');
    lgraph = connectLayers(lgraph, 'conv_2', 'lrelu_2');
    lastLayerName = 'lrelu_2';
    
    % Layer 3: Downsample 4x
    lgraph = addLayers(lgraph, convolution1dLayer(41, 256, 'Stride', 4, 'Padding', 'same', 'Name', 'conv_3'));
    lgraph = addLayers(lgraph, leakyReluLayer(0.2, 'Name', 'lrelu_3'));
    lgraph = connectLayers(lgraph, lastLayerName, 'conv_3');
    lgraph = connectLayers(lgraph, 'conv_3', 'lrelu_3');
    lastLayerName = 'lrelu_3';
    
    % Layer 4: Downsample 4x
    lgraph = addLayers(lgraph, convolution1dLayer(41, 512, 'Stride', 4, 'Padding', 'same', 'Name', 'conv_4'));
    lgraph = addLayers(lgraph, leakyReluLayer(0.2, 'Name', 'lrelu_4'));
    lgraph = connectLayers(lgraph, lastLayerName, 'conv_4');
    lgraph = connectLayers(lgraph, 'conv_4', 'lrelu_4');
    lastLayerName = 'lrelu_4';
    
    % Layer 5: Conv Biasa (Feature Extraction)
    lgraph = addLayers(lgraph, convolution1dLayer(5, 1024, 'Padding', 'same', 'Name', 'conv_5'));
    lgraph = addLayers(lgraph, leakyReluLayer(0.2, 'Name', 'lrelu_5'));
    lgraph = connectLayers(lgraph, lastLayerName, 'conv_5');
    lgraph = connectLayers(lgraph, 'conv_5', 'lrelu_5');
    lastLayerName = 'lrelu_5';
    
    % Layer 6: Final Projection (Ke 1 Channel)
    lgraph = addLayers(lgraph, convolution1dLayer(3, 1, 'Padding', 'same', 'Name', 'final_conv'));
    lgraph = connectLayers(lgraph, lastLayerName, 'final_conv');
    
    % --- REVISI: GLOBAL AVERAGE POOLING ---
    % Mengubah input [Time x 1] menjadi [1 x 1] (Rata-rata seluruh waktu)
    lgraph = addLayers(lgraph, globalAveragePooling1dLayer('Name', 'global_pool'));
    lgraph = connectLayers(lgraph, 'final_conv', 'global_pool');
end