using llama.torchsharp;

namespace llama.cs;

enum TokenizerType
{
    llama_cs,
    llama_3,
}

static class Entrypoint
{
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

        var isBinarySerialized = modelWeightPath.EndsWith (".bin");

        (transformer.config, transformer.weights) = isBinarySerialized
            ? BinModelLoader.load (modelWeightPath)
            : PickleModelLoader.load (
                paramJsonPath: paramJsonPath,
                modelWeightPath: modelWeightPath);
        transformer.state = RunState.createRunState (transformer.config);

        if (steps == 0 || steps > transformer.config.seq_len) {
            steps = transformer.config.seq_len;
        }

        ITokenizer tokenizer = tokenizerType switch {
            TokenizerType.llama_cs => Tokenizer.create (tokenizerModelPath, transformer.config.vocab_size),
            TokenizerType.llama_3 => new torchsharp.tokenizer.llama3.Llama3Tokenizer (tokenizerModelPath),
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
