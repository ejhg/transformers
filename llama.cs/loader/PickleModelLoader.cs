using llama.cs.reference;
using llama.torchsharp;
using System.Text.Json;

namespace llama.cs.svd;

public class PickleModelLoader
{
    public static (config config, weights weights) load (
        string paramJsonPath,
        string modelWeightPath,
        bool useSVD = true
    ) {
        var json = JsonSerializer.Deserialize<ConfigurationParamsJson> (File.ReadAllText (paramJsonPath));

        using var fileStream = File.OpenRead (modelWeightPath);
        var hashtable = DelayedExecutionUnpickler.UnpickleStateDict (fileStream, leaveOpen: true);

        var lookup = hashtable.Keys.Cast<string> ().ToArray ().Select (key => {
            // FIX: to support stories-260k
            var _key = key.StartsWith ("_orig_mod.")
                ? key.Replace ("_orig_mod.", "")
                : key;
            return (_key, hashtable[key]);
        }).ToDictionary ();

        (int[] shape, float[] floats) unpickle (string key) {
            Console.WriteLine (key);
            return PickleLoader.readTensor (((Func<object[]>)lookup[key]) (), key);
        }

        var hiddenDim = (int)((json.ffn_dim_multiplier ?? 1) * 2 * (json.dim * 4) / 3);
        // Round the hidden_dim to the nearest multiple of the multiple_of parameter
        hiddenDim = json.multiple_of * ((hiddenDim + json.multiple_of - 1) / json.multiple_of);

        var tok_embeddings = unpickle ("tok_embeddings.weight");

        var p = new config {
            dim = json.dim,
            hidden_dim = hiddenDim, // 172
            n_layers = json.n_layers,
            n_heads = json.n_heads,
            n_kv_heads = json.n_kv_heads ?? json.n_heads,
            vocab_size = tok_embeddings.shape[0],
            seq_len = 256, // TODO
        };

        // Map weights
        var head_size = p.dim / p.n_heads;
        var n_layers = p.n_layers;

        var w = new weights ();

        // Token embedding table
        {
            var index = 0;
            w.token_embedding_table = new float[p.vocab_size][];
            var token_embedding_weight = tok_embeddings.floats;

            for (var i = 0; i < p.vocab_size; i++) {
                w.token_embedding_table[i] = new float[p.dim];
                for (var j = 0; j < p.dim; j++) {
                    w.token_embedding_table[i][j] = token_embedding_weight[index++];
                }
            }
        }

        w.layers = new weights.LayerWeights[n_layers];

        for (var l = 0; l < n_layers; l++) {
            w.layers[l] = new weights.LayerWeights {
                rms_att_weight = new float[p.dim],
                wq = new float[p.n_heads * head_size, p.dim],
                wk = new float[p.n_kv_heads * head_size, p.dim],
                wv = new float[p.n_kv_heads * head_size, p.dim],
                wo = new float[p.dim, p.n_heads * head_size],
                rms_ffn_weight = new float[p.dim],
                w1 = new float[p.hidden_dim, p.dim],
                w2 = new float[p.dim, p.hidden_dim],
                w3 = new float[p.hidden_dim, p.dim],
                // Initialize SVD matrices as not-used by default
                wq_svd = new weights.SVDMatrix { use_svd = false },
                wk_svd = new weights.SVDMatrix { use_svd = false },
                wv_svd = new weights.SVDMatrix { use_svd = false },
                wo_svd = new weights.SVDMatrix { use_svd = false },
                w1_svd = new weights.SVDMatrix { use_svd = false },
                w2_svd = new weights.SVDMatrix { use_svd = false },
                w3_svd = new weights.SVDMatrix { use_svd = false }
            };
        }

        for (var l = 0; l < n_layers; l++) {
            // RMSNorm weights
            {
                var data = unpickle ($"layers.{l}.attention_norm.weight").floats;
                var index = 0;
                for (var j = 0; j < p.dim; j++) {
                    w.layers[l].rms_att_weight[j] = data[index++];
                }
            }

            // wq weights
            {
                var data = unpickle ($"layers.{l}.attention.wq.weight").floats;
                var index = 0;
                for (var i = 0; i < p.n_heads * head_size; i++) {
                    for (var j = 0; j < p.dim; j++)
                        w.layers[l].wq[i, j] = data[index++];
                }
            }

            // wk weights
            {
                var data = unpickle ($"layers.{l}.attention.wk.weight").floats;
                var index = 0;
                for (var i = 0; i < p.n_kv_heads * head_size; i++) {
                    for (var j = 0; j < p.dim; j++)
                        w.layers[l].wk[i, j] = data[index++];
                }
            }

            // wv weights
            {
                var data = unpickle ($"layers.{l}.attention.wv.weight").floats;
                var index = 0;
                for (var i = 0; i < p.n_kv_heads * head_size; i++) {
                    for (var j = 0; j < p.dim; j++)
                        w.layers[l].wv[i, j] = data[index++];
                }
            }

            // wo weights
            {
                var data = unpickle ($"layers.{l}.attention.wo.weight").floats;
                var index = 0;
                for (var i = 0; i < p.dim; i++) {
                    for (var j = 0; j < p.n_heads * head_size; j++)
                        w.layers[l].wo[i, j] = data[index++];
                }
            }

            // RMSNorm FFN weights
            {
                var data = unpickle ($"layers.{l}.ffn_norm.weight").floats;
                var index = 0;
                for (var j = 0; j < p.dim; j++) {
                    w.layers[l].rms_ffn_weight[j] = data[index++];
                }
            }

            // w1 weights
            {
                var unpickled = unpickle ($"layers.{l}.feed_forward.w1.weight");
                var data = unpickled.floats;
                var index = 0;

                if (data.Length != p.hidden_dim * p.dim) {
                    throw new Exception ();
                }

                for (var i = 0; i < p.hidden_dim; i++) {
                    for (var j = 0; j < p.dim; j++)
                        w.layers[l].w1[i, j] = data[index++];
                }
            }

            // w2 weights
            {
                var data = unpickle ($"layers.{l}.feed_forward.w2.weight").floats;
                var index = 0;

                if (data.Length != p.hidden_dim * p.dim) {
                    throw new Exception ();
                }

                for (var i = 0; i < p.dim; i++) {
                    for (var j = 0; j < p.hidden_dim; j++)
                        w.layers[l].w2[i, j] = data[index++];
                }
            }

            // w3 weights
            {
                var data = unpickle ($"layers.{l}.feed_forward.w3.weight").floats;
                var index = 0;

                if (data.Length != p.hidden_dim * p.dim) {
                    throw new Exception ();
                }

                for (var i = 0; i < p.hidden_dim; i++) {
                    for (var j = 0; j < p.dim; j++)
                        w.layers[l].w3[i, j] = data[index++];
                }
            }
        }

        // Final RMSNorm weights
        {
            var index = 0;
            w.rms_final_weight = new float[p.dim];
            var data = unpickle ("norm.weight").floats;

            if (data.Length != p.dim) {
                throw new Exception ();
            }

            for (var i = 0; i < p.dim; i++) {
                w.rms_final_weight[i] = data[index++];
            }
        }

        {
            var index = 0;
            w.wcls = new float[p.vocab_size][];
            var data = unpickle ("output.weight").floats;

            if (data.Length != p.dim * p.vocab_size) {
                throw new Exception ();
            }

            for (var i = 0; i < p.vocab_size; i++) {
                w.wcls[i] = new float[p.dim];
                for (var j = 0; j < p.dim; j++)
                    w.wcls[i][j] = data[index++];
            }
        }

        fileStream.Close ();

        // Apply SVD decomposition to large matrices if requested
        if (useSVD) {
            Console.WriteLine("Applying SVD decomposition to model weights...");
            ModelSVDDecomposer.DecomposeMatrices(w, p);
        }

        return (p, w);
    }
}
