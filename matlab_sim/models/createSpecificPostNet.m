function lgraph = createSpecificPostNet(inputChannels)
    % PostNet sederhana: 5 Layer Conv1D -> Tanh
    % InputChannels biasanya 128 (dari output Generator sebelum proyeksi akhir)
    
    lgraph = layerGraph();
    lgraph = addLayers(lgraph, sequenceInputLayer(inputChannels, 'Name', 'post_in', 'Normalization', 'none'));
    
    currentName = 'post_in';
    numLayers = 5;
    numCh = 512; % Channel internal PostNet biasanya besar
    
    for i = 1:(numLayers-1)
        convName = sprintf('post_conv_%d', i);
        bnName   = sprintf('post_bn_%d', i);
        actName  = sprintf('post_tanh_%d', i);
        
        % Conv -> BN -> Tanh
        lgraph = addLayers(lgraph, convolution1dLayer(5, numCh, 'Padding', 'same', 'Name', convName));
        lgraph = addLayers(lgraph, batchNormalizationLayer('Name', bnName));
        lgraph = addLayers(lgraph, tanhLayer('Name', actName));
        
        lgraph = connectLayers(lgraph, currentName, convName);
        lgraph = connectLayers(lgraph, convName, bnName);
        lgraph = connectLayers(lgraph, bnName, actName);
        
        currentName = actName;
    end
    
    % Layer Terakhir (Proyeksi ke Audio Waveform 1 Channel)
    lgraph = addLayers(lgraph, convolution1dLayer(5, 1, 'Padding', 'same', 'Name', 'post_final_conv'));
    lgraph = addLayers(lgraph, tanhLayer('Name', 'post_out')); % Audio range -1 s/d 1
    
    lgraph = connectLayers(lgraph, currentName, 'post_final_conv');
    lgraph = connectLayers(lgraph, 'post_final_conv', 'post_out');
end