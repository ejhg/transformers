namespace llama.cs;

public class ModelSVDDecomposer
{
    public static void DecomposeMatrices(weights weights, config config) {
        Console.WriteLine("Starting SVD decomposition of model matrices...");

        // Decompose each layer's matrices in parallel for faster loading
        Parallel.For(0, config.n_layers, l => {
            var layer = weights.layers[l];

            // Decompose attention matrices
            DecomposeMatrix(layer.wq, ref layer.wq_svd, SvdConstants.RankRatios.WQ, $"Layer {l} wq");
            DecomposeMatrix(layer.wk, ref layer.wk_svd, SvdConstants.RankRatios.WK, $"Layer {l} wk");
            DecomposeMatrix(layer.wv, ref layer.wv_svd, SvdConstants.RankRatios.WV, $"Layer {l} wv");
            DecomposeMatrix(layer.wo, ref layer.wo_svd, SvdConstants.RankRatios.WO, $"Layer {l} wo");

            // Decompose FFN matrices (these are typically the largest matrices in the model)
            // DecomposeMatrix(layer.w1, ref layer.w1_svd, SvdConstants.RankRatios.W1, $"Layer {l} w1");
            // DecomposeMatrix(layer.w2, ref layer.w2_svd, SvdConstants.RankRatios.W2, $"Layer {l} w2");
            // DecomposeMatrix(layer.w3, ref layer.w3_svd, SvdConstants.RankRatios.W3, $"Layer {l} w3");
        });

        Console.WriteLine("SVD decomposition completed.");
    }

    private static void DecomposeMatrix(float[,] matrix, ref weights.SVDMatrix svdMatrix, float rankRatio, string matrixName) {
        var rows = matrix.GetLength(0);
        var cols = matrix.GetLength(1);

        // Always set the original matrix
        svdMatrix.original = matrix;

        // Check if this matrix should use SVD
        // if (!SvdConstants.ShouldUseSvd(rows, cols, rankRatio)) {
        //     // SVD not beneficial for this matrix
        //     svdMatrix.use_svd = false;
        //     return;
        // }

        // Special handling for QKV matrices with 2D output shapes
        bool isAttentionQKV = matrixName.Contains("wq") || matrixName.Contains("wk") || matrixName.Contains("wv");

        // Log additional details for QKV matrices
        if (isAttentionQKV) {
            Console.WriteLine($"  Attention matrix {matrixName} will use special 2D output handling in MatMul2D");
            // Note: The actual reshaping happens in the MatMul2D method at inference time
        }

        // Calculate effective rank based on matrix dimensions and rank ratio
        var minDim = Math.Min(rows, cols);
        var effectiveRank = (int)(minDim * rankRatio);
        effectiveRank = Math.Max(effectiveRank, 1); // At least rank 1

        Console.WriteLine($"Decomposing {matrixName} [{rows}x{cols}] with rank {effectiveRank}");

        // Initialize SVD matrix
        svdMatrix = new weights.SVDMatrix {
            use_svd = true,
            rank = effectiveRank,
            original = matrix // Keep original for error calculation
        };

        // Perform the low-rank SVD decomposition
        LowRankSVD.LowRankDecompose(
            matrix,
            effectiveRank,
            out svdMatrix.U,
            out svdMatrix.S,
            out svdMatrix.Vt
        );

        // Calculate error ratio to verify decomposition quality
        var error = LowRankSVD.FrobeniusNorm(matrix, svdMatrix.U, svdMatrix.S, svdMatrix.Vt);
        var matrixNorm = 0.0f;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                matrixNorm += matrix[i, j] * matrix[i, j];
            }
        }
        matrixNorm = MathF.Sqrt(matrixNorm);

        svdMatrix.error_ratio = error / matrixNorm;

        // If error ratio is too high, don't use SVD for this matrix
        if (svdMatrix.error_ratio > SvdConstants.ErrorThreshold) {
            Console.WriteLine($"  Error ratio too high: {svdMatrix.error_ratio:F4}");
            // svdMatrix.use_svd = false;
            // return;
        }

        // Calculate memory savings
        long originalSize = rows * cols * sizeof(float);
        long svdSize = (rows * effectiveRank + effectiveRank + effectiveRank * cols) * sizeof(float);
        float compressionRatio = (float)originalSize / svdSize;

        Console.WriteLine($"  Decomposition complete: error ratio {svdMatrix.error_ratio:F4}, compression ratio {compressionRatio:F2}x");

        // Optionally, free the original matrix to save memory
        // svdMatrix.original = null;
    }
}
