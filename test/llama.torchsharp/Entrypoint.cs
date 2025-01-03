﻿using llama.torchsharp.tokenizer.llama2;
using llama.torchsharp.tokenizer.llama3;
using TorchSharp;

namespace llama.torchsharp;

static class Entrypoint
{
    public static void run (
        string weightsDir,
        string llamaResourcesDir,
        string weightsFile,
        string prompt
    ) {
        var device = "cpu";

        torch.manual_seed (100);

        Console.WriteLine ("running on " + device);

        ITokenizer tokenizer = File.Exists ($"{weightsDir}/tokenizer.model")
            ? new Llama3Tokenizer (
                $"{weightsDir}/tokenizer.model")
            : new Llama2Tokenizer (
                $"{llamaResourcesDir}/vocab.json",
                $"{llamaResourcesDir}/merges.txt");

        var model = Model.build (
            modelFolder: weightsDir,
            tokenizer: tokenizer,
            maxSeqLen: 128,
            maxBatchSize: 1,
            device: device,
            modelWeightPath: weightsFile);

        var prompts = new[] {
            prompt,
        };

        var result = Inference.TextCompletion (
            model,
            tokenizer,
            prompts,
            temperature: 0,
            echo: true,
            device: device);

        foreach (var item in result) {
            Console.WriteLine ($"generation: {item.generation}");
        }
    }
}
