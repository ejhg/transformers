using System.Text;

namespace llama.cs;

/**
 * The Byte Pair Encoding (BPE) Tokenizer that translates strings <-> tokens
 */
public class Tokenizer
{
    string[] vocab;
    float[] vocab_scores;
    Dictionary<string, int> vocab_lookup;
    string[] byte_pieces = new string[512];

    public void BuildTokenizer (string tokenizer_path, int vocab_size) {
        vocab = new string[vocab_size];
        vocab_scores = new float[vocab_size];
        vocab_lookup = null;

        // Initialize byte pieces
        for (int i = 0; i < 256; i++) {
            byte_pieces[i * 2] = ((char)i).ToString ();
            byte_pieces[i * 2 + 1] = '\0'.ToString ();
        }

        using FileStream fs = new FileStream (tokenizer_path, FileMode.Open, FileAccess.Read);
        using BinaryReader br = new BinaryReader (fs);

        // variable not used, but must be read from structure
        var max_token_length = br.ReadUInt32 ();

        for (int i = 0; i < vocab_size; i++) {
            vocab_scores[i] = br.ReadSingle ();
            var len = br.ReadInt32 ();
            vocab[i] = Encoding.UTF8.GetString (br.ReadBytes (len));
        }
    }

    public string Decode (int prev_token, int token) {
        string piece = vocab[token];
        if (prev_token == 1 && piece.StartsWith (" ")) {
            piece = piece.Substring (1);
        }

        if (piece.StartsWith ("<0x") && piece.EndsWith (">")) {
            string hex = piece.Substring (3, piece.Length - 4);
            if (byte.TryParse (hex, System.Globalization.NumberStyles.HexNumber, null, out byte byte_val)) {
                piece = ((char)byte_val).ToString ();
            }
        }

        return piece;
    }

    public void SafePrint (string piece) {
        if (string.IsNullOrEmpty (piece) || piece.Length == 1 && char.IsControl (piece[0]) && !char.IsWhiteSpace (piece[0])) {
            return;
        }

        Console.Write (piece);
    }

    int StrLookup (string str) {
        if (vocab_lookup == null) {
            vocab_lookup = vocab
                .Select ((str, index) => (str, index))
                .ToDictionary ();
        }

        return vocab_lookup.GetValueOrDefault (str, -1);
    }

    public void Encode (string text, bool bos, bool eos, List<int> tokens) {
        if (text == null) {
            Console.Error.WriteLine ("Cannot encode null text");
            Environment.Exit (1);
        }

        // Start encoding
        if (bos)
            tokens.Add (1); // BOS token

        if (text.Length > 0) {
            int dummy_prefix = StrLookup (" ");
            tokens.Add (dummy_prefix);
        }

        int str_len = 0;
        StringBuilder str_buffer = new StringBuilder ();

        for (int i = 0; i < text.Length; i++) {
            char c = text[i];

            if ((c & 0xC0) != 0x80) {
                str_len = 0;
            }

            str_buffer.Append (c);
            str_len++;

            if (i + 1 < text.Length && (text[i + 1] & 0xC0) == 0x80 && str_len < 4) {
                continue;
            }

            string str = str_buffer.ToString ();
            int id = StrLookup (str);

            if (id != -1) {
                tokens.Add (id);
            } else {
                foreach (char ch in str) {
                    tokens.Add ((byte)ch + 3);
                }
            }

            str_buffer.Clear ();
            str_len = 0;
        }

        // Merge pairs
        while (true) {
            float best_score = -1e10f;
            int best_id = -1;
            int best_idx = -1;

            for (int i = 0; i < tokens.Count - 1; i++) {
                string merged = vocab[tokens[i]] + vocab[tokens[i + 1]];
                int id = StrLookup (merged);
                if (id != -1 && vocab_scores[id] > best_score) {
                    best_score = vocab_scores[id];
                    best_id = id;
                    best_idx = i;
                }
            }

            if (best_idx == -1)
                break;

            tokens[best_idx] = best_id;
            tokens.RemoveAt (best_idx + 1);
        }

        if (eos)
            tokens.Add (2); // EOS token
    }
}
