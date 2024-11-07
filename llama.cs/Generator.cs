namespace llama.cs;

public class Generator
{
    public static void Generate (Transformer transformer, Tokenizer tokenizer, Sampler sampler, string prompt, int steps) {
        if (prompt == null) {
            prompt = "";
        }

        List<int> tokens = new List<int> ();
        tokenizer.Encode (prompt, true, false, tokens);

        if (tokens.Count < 1) {
            Console.Error.WriteLine ("Expected at least 1 prompt token");
            Environment.Exit (1);
        }

        long start = 0;
        int next = 0;
        int token = tokens[0];
        int pos = 0;

        while (pos < steps) {
            float[] logits = TransformerModel.Forward (transformer, token, pos);

            if (pos < tokens.Count - 1) {
                next = tokens[pos + 1];
            } else {
                next = sampler.Sample (logits);
            }

            pos++;

            if (next == 1) break;

            string piece = tokenizer.Decode (token, next);
            tokenizer.SafePrint (piece);
            token = next;

            if (start == 0) start = TimeInMs ();
        }

        Console.WriteLine ();

        if (pos > 1) {
            long end = TimeInMs ();
            Console.Error.WriteLine ($"Achieved tok/s: {(pos - 1) / ((end - start) / 1000.0)}");
        }
    }

    static long TimeInMs () {
        return DateTimeOffset.Now.ToUnixTimeMilliseconds ();
    }

    public static void Chat (Transformer transformer, Tokenizer tokenizer, Sampler sampler, string cli_user_prompt, string cli_system_prompt,
        int steps) {
        // Buffers for prompts
        string system_prompt = "";
        string user_prompt = "";
        string rendered_prompt = "";
        List<int> prompt_tokens = new List<int> ();
        int user_idx = 0;

        bool user_turn = true; // User starts
        int next = 0; // Next token
        int token = 0; // Current token
        int pos = 0; // Position in the sequence

        while (pos < steps) {
            // User's turn
            if (user_turn) {
                // Get system prompt at position 0
                if (pos == 0) {
                    if (cli_system_prompt == null) {
                        // System prompt not provided, read from stdin
                        ReadStdin ("Enter system prompt (optional): ", out system_prompt);
                    } else {
                        system_prompt = cli_system_prompt;
                    }
                }

                // Get user prompt
                if (pos == 0 && cli_user_prompt != null) {
                    user_prompt = cli_user_prompt;
                } else {
                    ReadStdin ("User: ", out user_prompt);
                }

                // Render prompts into the Llama 2 Chat schema
                if (pos == 0 && !string.IsNullOrEmpty (system_prompt)) {
                    rendered_prompt = $"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_prompt} [/INST]";
                } else {
                    rendered_prompt = $"[INST] {user_prompt} [/INST]";
                }

                // Encode the rendered prompt into tokens
                prompt_tokens.Clear ();
                tokenizer.Encode (rendered_prompt, true, false, prompt_tokens);
                user_idx = 0; // Reset user index
                user_turn = false;
                Console.Write ("Assistant: ");
            }

            // Determine the token to pass into the transformer next
            if (user_idx < prompt_tokens.Count) {
                token = prompt_tokens[user_idx++];
            } else {
                token = next;
            }

            // EOS token ends the Assistant turn
            if (token == 2) {
                user_turn = true;
            }

            // Forward the transformer to get logits
            float[] logits = TransformerModel.Forward (transformer, token, pos);
            next = sampler.Sample (logits);
            pos++;

            if (user_idx >= prompt_tokens.Count && next != 2) {
                // Assistant is responding
                string piece = tokenizer.Decode (token, next);
                tokenizer.SafePrint (piece);
            }

            if (next == 2) {
                Console.WriteLine ();
            }
        }

        Console.WriteLine ();
    }

    static void ReadStdin (string guide, out string buffer) {
        Console.Write (guide);
        buffer = Console.ReadLine ();
    }
}
