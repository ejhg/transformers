using Tiktoken;

namespace llama.torchsharp.tokenizer.llama3;

public class SentencePieceTokenizer : ITokenizer
{
    const int num_reserved_special_tokens = 256;

    const string pat_str = @"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

    CoreBpe bpe;

    public int VocabSize { get; }

    public int PadId { get; }

    public int EosId { get; }

    int BosId { get; }

    public SentencePieceTokenizer (string modelPath) {
        var model = SentencePieceModelParser.Parse (File.ReadAllBytes (modelPath));

        var mergeable_ranks = model.SentencePieces
            .OrderBy (_ => _.Score)
            .Select ((_, index) => (System.Text.Encoding.UTF8.GetBytes (_.Piece), index))
            .ToDictionary ();

        var num_base_tokens = mergeable_ranks.Count;

        var _special_tokens = new[] {
            "<|begin_of_text|>",
            "<|end_of_text|>",
            "<|reserved_special_token_0|>",
            "<|reserved_special_token_1|>",
            "<|finetune_right_pad_id|>",
            "<|step_id|>",
            "<|start_header_id|>",
            "<|end_header_id|>",
            "<|eom_id|>", // end of message
            "<|eot_id|>", // end of turn
            "<|python_tag|>",
        };

        var reserved_tokens = Enumerable
            .Range (0, num_reserved_special_tokens - _special_tokens.Length)
            .Select (i => $"<|reserved_special_token_{2 + i}|>")
            .ToArray ();

        var special_tokens = _special_tokens
            .Concat (reserved_tokens)
            .Select ((token, index) => (token, num_base_tokens + index))
            .ToDictionary ();
        bpe = new CoreBpe (mergeable_ranks, special_tokens, pat_str);

        VocabSize = num_base_tokens + special_tokens.Count;
        PadId = special_tokens["<|finetune_right_pad_id|>"];
        EosId = special_tokens["<|end_of_text|>"];
        BosId = special_tokens["<|begin_of_text|>"];
    }

    public int[] Encode (string input, bool bos, bool eos) {
        var tokens = bpe.EncodeNative (input, [], []);

        if (bos) {
            tokens = new[] { this.BosId }.Concat (tokens).ToArray ();
        }

        if (eos) {
            tokens = tokens.Concat (new[] { this.EosId }).ToArray ();
        }

        return tokens.ToArray ();
    }

    public string Decode (int[] input) {
        return System.Text.Encoding.UTF8.GetString (bpe.DecodeNative (input));
    }
}
