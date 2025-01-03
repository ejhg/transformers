var HOME = Environment.GetFolderPath (Environment.SpecialFolder.UserProfile);

var command = "llama.c:stories-260k";

Console.WriteLine ($"Running {command}");

switch (command) {
    case "llama.c:stories-260k":
        llama.c.Program.main (
            $"{HOME}/llama/stories-260k/stories260K.bin",
            "-z",
            $"{HOME}/llama/stories-260k/tok512.bin");
        break;

    case "llama.cs:stories-260k":
        llama.cs.Entrypoint.main (
            llama.cs.TokenizerType.llama_cs,
            tokenizerModelPath: $"{HOME}/llama/stories-260k/tok512.bin",
            paramJsonPath: $"{HOME}/llama/stories-260k/params.json",
            modelWeightPath: $"{HOME}/llama/stories-260k/stories260K.pth",
            rng_seed: 1,
            temperature: 0,
            prompt: "Once upon a time");
        break;

    case "llama.cs:stories-15m":
        llama.cs.Entrypoint.main (
            llama.cs.TokenizerType.llama_cs,
            tokenizerModelPath: "resources/tokenizer.bin",
            paramJsonPath: $"{HOME}/llama/stories-15m/params.json",
            modelWeightPath: $"{HOME}/llama/stories-15m/stories15M.pth",
            rng_seed: 1,
            temperature: 0,
            prompt: "Once upon a time");
        break;

    case "llama.cs:stories-15m.bin":
        llama.cs.Entrypoint.main (
            llama.cs.TokenizerType.llama_cs,
            tokenizerModelPath: "resources/tokenizer.bin",
            modelWeightPath: "resources/stories15M.bin",
            rng_seed: 1,
            prompt: "Once upon a time");
        break;

    case "llama.cs:llama-3.2-1b-instruct":
        llama.cs.Entrypoint.main (
            llama.cs.TokenizerType.llama_3,
            paramJsonPath: $"{HOME}/llama/llama-3.2-1b-instruct/params.json",
            modelWeightPath: $"{HOME}/llama/llama-3.2-1b-instruct/consolidated.00.pth",
            tokenizerModelPath: $"{HOME}/llama/llama-3.2-1b-instruct/tokenizer.model",
            rng_seed: 1,
            temperature: 0,
            prompt: "The meaning of life is ");
        break;

    case "llama.cs:llama-3.2-1b-instruct.bin":
        llama.cs.Entrypoint.main (
            llama.cs.TokenizerType.llama_3,
            paramJsonPath: $"{HOME}/llama/llama-3.2-1b-instruct/params.json",
            modelWeightPath: $"{HOME}/llama/llama-3.2-1b-instruct/llama-3.2-1b-instruct.bin",
            tokenizerModelPath: $"{HOME}/llama/llama-3.2-1b-instruct/tokenizer.model",
            rng_seed: 1,
            temperature: 0,
            prompt: "The meaning of life is ");
        break;

    case "llama.torchsharp:llama-3.2-1b-instruct":
        llama.torchsharp.Entrypoint.run (
            HOME + "/llama/llama-3.2-1b-instruct",
            "llama.torchsharp/resources",
            weightsFile: "consolidated.00.pth",
            prompt: "I believe the meaning of life is");
        break;

    case "llama.torchsharp:stories-15m":
        llama.torchsharp.Entrypoint.run (
            HOME + "/llama/stories-15m",
            "llama.torchsharp/resources", // TODO
            weightsFile: "stories15M.pth",
            prompt: "Once upon a time");
        break;

    default:
        throw new NotSupportedException (command);
}

Console.WriteLine ("done");
