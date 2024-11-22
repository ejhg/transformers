namespace llama.cs;

class Program
{
    static void ErrorUsage () {
        Console.WriteLine ("Usage:   run <checkpoint> [options]");
        Console.WriteLine ("Example: run model.bin -n 256 -i \"Once upon a time\"");
        Console.WriteLine ("Options:");
        Console.WriteLine ("  -t <float>  temperature in [0,inf], default 1.0");
        Console.WriteLine ("  -p <float>  p value in top-p (nucleus) sampling in [0,1] default 0.9");
        Console.WriteLine ("  -s <int>    random seed, default time(NULL)");
        Console.WriteLine ("  -n <int>    number of steps to run for, default 256. 0 = max_seq_len");
        Console.WriteLine ("  -i <string> input prompt");
        Console.WriteLine ("  -z <string> optional path to custom tokenizer");
        Console.WriteLine ("  -m <string> mode: generate|chat, default: generate");
        Console.WriteLine ("  -y <string> (optional) system prompt in chat mode");
        Environment.Exit (1);
    }

    public static void main (params string[] args) {
        // Default parameters
        string checkpoint_path = null;
        string tokenizer_path = "resources/tokenizer.bin";
        float temperature = 1.0f;
        float topp = 0.9f;
        int steps = 256;
        string prompt = null;
        int rng_seed = 0;
        string mode = "generate";
        string system_prompt = null;

        if (args.Length >= 1) {
            checkpoint_path = args[0];
        } else {
            ErrorUsage ();
        }

        // Argument parsing
        for (int i = 1; i < args.Length; i += 2) {
            if (i + 1 >= args.Length) ErrorUsage ();
            if (args[i][0] != '-') ErrorUsage ();
            if (args[i].Length != 2) ErrorUsage ();

            switch (args[i][1]) {
                case 't':
                    temperature = float.Parse (args[i + 1]);
                    break;
                case 'p':
                    topp = float.Parse (args[i + 1]);
                    break;
                case 's':
                    rng_seed = int.Parse (args[i + 1]);
                    break;
                case 'n':
                    steps = int.Parse (args[i + 1]);
                    break;
                case 'i':
                    prompt = args[i + 1];
                    break;
                case 'z':
                    tokenizer_path = args[i + 1];
                    break;
                case 'm':
                    mode = args[i + 1];
                    break;
                case 'y':
                    system_prompt = args[i + 1];
                    break;
                default:
                    ErrorUsage ();
                    break;
            }
        }

        if (rng_seed <= 0) rng_seed = (int)DateTime.Now.Ticks;
        if (temperature < 0.0) temperature = 0.0f;
        if (topp < 0.0 || topp > 1.0) topp = 0.9f;
        if (steps < 0) steps = 0;

        // Build the Transformer via the model .bin file
        var transformer = new model ();
        (transformer.config, transformer.weights) = ModelLoader.ReadCheckpoint (checkpoint_path);
        transformer.state = ModelLoader.createRunState (transformer.config);

        if (steps == 0 || steps > transformer.config.seq_len) {
            steps = transformer.config.seq_len;
        }

        // Build the Tokenizer via the tokenizer .bin file
        Tokenizer tokenizer = new Tokenizer ();
        tokenizer.BuildTokenizer (tokenizer_path, transformer.config.vocab_size);

        // Build the Sampler
        Sampler sampler = new Sampler (transformer.config.vocab_size, temperature, topp, rng_seed);

        // Run!
        if (mode == "generate") {
            Generator.Generate (transformer, tokenizer, sampler, prompt, steps);
        } else if (mode == "chat") {
            Generator.Chat (transformer, tokenizer, sampler, prompt, system_prompt, steps);
        } else {
            Console.Error.WriteLine ($"Unknown mode: {mode}");
            ErrorUsage ();
        }
    }
}
