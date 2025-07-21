using llama.torchsharp;
using llama.torchsharp.tokenizer.llama2;
using llama.torchsharp.tokenizer.llama3;
using TorchSharp;

namespace transformers.entrypoints;

public class llama_touchsharp
{
    public static void run (string command) {
        var HOME = Environment.GetFolderPath (Environment.SpecialFolder.UserProfile);

        switch (command) {
            case "llama.torchsharp:llama-3.2-1b-instruct":
                run (
                    HOME + "/llama/llama-3.2-1b-instruct",
                    "llama.torchsharp/resources",
                    weightsFile: "consolidated.00.pth",
                    prompt: "I believe the meaning of life is");
                break;

            case "llama.torchsharp:stories-15m":
                run (
                    HOME + "/llama/stories-15m",
                    "llama.torchsharp/resources", // TODO
                    weightsFile: "stories15M.pth",
                    prompt: "Once upon a time");
                break;

            default:
                throw new NotSupportedException (command);
        }

        Console.WriteLine ("done");
    }

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
