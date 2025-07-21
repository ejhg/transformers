using static TorchSharp.torch;

namespace mingpt.torchsharp;

public static class Utils
{
    public static Configuration GetDefaultConfig () {
        return new Configuration {
            vocab_size = 50257, // GPT-2 vocab size
            block_size = 1024,
            n_layer = 12,
            n_head = 12,
            n_embd = 768,
            embd_pdrop = 0.1f,
            resid_pdrop = 0.1f,
            attn_pdrop = 0.1f,
            dtype = ScalarType.Float32
        };
    }

    public static TrainerConfig GetDefaultTrainerConfig () {
        return new TrainerConfig {
            device = null,
            num_workers = 4,
            max_iters = 1000,
            batch_size = 32,
            learning_rate = 3e-4,
            betas = (0.9, 0.95),
            weight_decay = 0.1,
            grad_norm_clip = 1.0
        };
    }
}

// Simple text dataset for character-level language modeling
public class CharDataset : utils.data.Dataset<Dictionary<string, Tensor>>
{
    private readonly string text;
    private readonly Dictionary<char, int> stoi;
    private readonly Dictionary<int, char> itos;
    private readonly int block_size;

    public CharDataset (string text, int block_size) {
        this.text = text;
        this.block_size = block_size;

        var chars = text.ToCharArray ().Distinct ().OrderBy (c => c).ToList ();
        this.stoi = chars.Select ((c, i) => new {
            c,
            i
        }).ToDictionary (x => x.c, x => x.i);
        this.itos = this.stoi.ToDictionary (kv => kv.Value, kv => kv.Key);
    }

    public int VocabSize => this.stoi.Count;

    public override long Count => this.text.Length - this.block_size;

    public override Dictionary<string, Tensor> GetTensor (long index) {
        var chunk = this.text.Substring ((int)index, this.block_size + 1);
        var dix = chunk.Select (c => this.stoi[c]).ToArray ();

        var x = tensor (dix.Take (this.block_size).ToArray (), dtype: ScalarType.Int64);
        var y = tensor (dix.Skip (1).ToArray (), dtype: ScalarType.Int64);

        return new Dictionary<string, Tensor> {
            ["input_ids"] = x,
            ["labels"] = y
        };
    }

    public string Decode (int[] indices) {
        return new string (indices.Select (i => this.itos[i]).ToArray ());
    }

    public int[] Encode (string text) {
        return text.Select (c => this.stoi[c]).ToArray ();
    }
}
