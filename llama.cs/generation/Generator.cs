using llama.torchsharp;

namespace llama.cs;

public class Generator
{
    public static void Generate (model transformer, ITokenizer tokenizer, Sampler sampler, string prompt, int steps) {
        if (prompt == null) {
            throw new ArgumentException ("prompt is null");
        }

        var tokens = tokenizer.Encode (prompt, true, false).ToList ();

        if (tokens.Count < 1) {
            throw new Exception ("Expected at least 1 prompt token");
        }

        var start = System.Diagnostics.Stopwatch.StartNew ();
        int pos = 0;

        float[] logits = null;

        foreach (var curr in tokens) {
            logits = ForwardPass.Forward (transformer, curr, pos++);
        }

        var next = sampler.Sample (logits);

        while (pos < steps) {
            logits = ForwardPass.Forward (transformer, next, pos++);
            next = sampler.Sample (logits);

            if (next == 1) {
                break;
            }

            tokens.Add (next);

            var piece = tokenizer.Decode (tokens.ToArray ());
            Console.WriteLine (piece);
        }

        Console.WriteLine ();

        if (pos > 1) {
            var elapsed = start.ElapsedMilliseconds;
            Console.Error.WriteLine ($"Achieved tok/s: {(pos - 1) / (elapsed / 1000.0)}");
        }
    }

    public static void Chat (
        model transformer,
        ITokenizer tokenizer,
        Sampler sampler,
        string cli_user_prompt,
        string cli_system_prompt,
        int steps
    ) {
        // Buffers for prompts
        string system_prompt = "";
        string user_prompt = "";
        string rendered_prompt = "";
        var prompt_tokens = Array.Empty<int>();
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
                prompt_tokens = tokenizer.Encode (rendered_prompt, true, false);
                user_idx = 0; // Reset user index
                user_turn = false;
                Console.Write ("Assistant: ");
            }

            // Determine the token to pass into the transformer next
            if (user_idx < prompt_tokens.Length) {
                token = prompt_tokens[user_idx++];
            } else {
                token = next;
            }

            // EOS token ends the Assistant turn
            if (token == 2) {
                user_turn = true;
            }

            // Forward the transformer to get logits
            float[] logits = ForwardPass.Forward (transformer, token, pos);
            next = sampler.Sample (logits);
            pos++;

            if (user_idx >= prompt_tokens.Length && next != 2) {
                // Assistant is responding
                var piece = tokenizer.Decode ([token, next]);
                Console.Write (piece);
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
