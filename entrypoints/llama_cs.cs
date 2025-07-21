using llama.cs.reference;
using llama.torchsharp;
using llama.torchsharp.tokenizer.llama3;

namespace transformers.entrypoints;

public class llama_cs
{
    public enum TokenizerType
    {
        llama_cs,
        llama_3,
        sentencepiece,
    }

    public static void run (string command) {
        var HOME = Environment.GetFolderPath (Environment.SpecialFolder.UserProfile);

        switch (command) {
            case "llama.cs:stories-260k":
                main (
                    TokenizerType.llama_cs,
                    tokenizerModelPath: $"{HOME}/llama/stories-260k/tok512.bin",
                    paramJsonPath: $"{HOME}/llama/stories-260k/params.json",
                    modelWeightPath: $"{HOME}/llama/stories-260k/stories260K.pth",
                    rng_seed: 1,
                    temperature: 0,
                    prompt: "Once upon a time");
                break;

            case "llama.cs:stories-15m":
                main (
                    TokenizerType.llama_cs,
                    tokenizerModelPath: "resources/tokenizer.bin",
                    paramJsonPath: $"{HOME}/llama/stories-15m/params.json",
                    modelWeightPath: $"{HOME}/llama/stories-15m/stories15M.pth",
                    rng_seed: 1,
                    temperature: 0,
                    prompt: "Once upon a time");
                break;

            case "llama.cs:stories-15m.bin":
                main (
                    TokenizerType.llama_cs,
                    tokenizerModelPath: "resources/tokenizer.bin",
                    modelWeightPath: "resources/stories15M.bin",
                    rng_seed: 1,
                    prompt: "Once upon a time");
                break;

            case "llama.cs:llama-3.2-1b-instruct":
                main (
                    TokenizerType.llama_3,
                    paramJsonPath: $"{HOME}/llama/llama-3.2-1b-instruct/params.json",
                    modelWeightPath: $"{HOME}/llama/llama-3.2-1b-instruct/consolidated.00.pth",
                    tokenizerModelPath: $"{HOME}/llama/llama-3.2-1b-instruct/tokenizer.model",
                    rng_seed: 1,
                    temperature: 0,
                    prompt: "The meaning of life is ");
                break;

            case "llama.cs:llama-3.2-1b-instruct.bin":
                main (
                    TokenizerType.llama_3,
                    paramJsonPath: $"{HOME}/llama/llama-3.2-1b-instruct/params.json",
                    modelWeightPath: $"{HOME}/llama/llama-3.2-1b-instruct/llama-3.2-1b-instruct.bin",
                    tokenizerModelPath: $"{HOME}/llama/llama-3.2-1b-instruct/tokenizer.model",
                    rng_seed: 1,
                    temperature: 0,
                    prompt: "The meaning of life is ");
                break;

            default:
                throw new NotSupportedException (command);
        }

        Console.WriteLine ("done");
    }

    public static void main (
        TokenizerType tokenizerType,
        string tokenizerModelPath,
        string modelWeightPath,
        string paramJsonPath = null,
        float temperature = 1.0f,
        float topp = 0.9f,
        int steps = 256,
        string prompt = null,
        int? rng_seed = null,
        string mode = "generate",
        string system_prompt = null
    ) {
        if (temperature < 0.0) temperature = 0.0f;
        if (topp < 0.0 || topp > 1.0) topp = 0.9f;
        if (steps < 0) steps = 0;

        var transformer = new model ();

        (transformer.config, transformer.weights) = modelWeightPath.EndsWith (".bin")
            ? BinModelLoader.load (modelWeightPath)
            : PickleModelLoader.load (
                paramJsonPath: paramJsonPath,
                modelWeightPath: modelWeightPath);
        transformer.state = RunState.createRunState (transformer.config);

        if (steps == 0 || steps > transformer.config.seq_len) {
            steps = transformer.config.seq_len;
        }

        ITokenizer tokenizer = tokenizerType switch {
            TokenizerType.sentencepiece => new SentencePieceTokenizer (tokenizerModelPath),
            TokenizerType.llama_cs => Tokenizer.fromBinary (tokenizerModelPath, transformer.config.vocab_size),
            TokenizerType.llama_3 => new Llama3Tokenizer (tokenizerModelPath),
        };

        var sampler = new Sampler (transformer.config.vocab_size, temperature, topp, rng_seed);

        if (mode == "generate") {
            Generator.Generate (transformer, tokenizer, sampler, prompt, steps);
        } else if (mode == "chat") {
            Generator.Chat (transformer, tokenizer, sampler, prompt, system_prompt, steps);
        } else {
            throw new NotSupportedException ($"Unknown mode: {mode}");
        }
    }
}
