namespace llama.cs;

public static class SvdConstants
{
    // Settings for SVD decomposition
    public const float ErrorThreshold = 0.01f; // Maximum acceptable error ratio
    
    // Target minimum dimensions for applying SVD
    // SVD makes more sense for larger matrices
    public const int MinRowsForSvd = 1024;  
    public const int MinColsForSvd = 1024;
    public const int MinElementsForSvd = 1024 * 1024; // 1M elements
    
    // Target minimum speed improvement ratio
    // Only apply SVD if we expect an improvement of at least 1.5x
    public const float MinSpeedupRatio = 1.5f;
    
    // Different matrices in the transformer have different characteristics
    // Here we define which ones are good candidates for SVD
    
    // The most computationally intensive matrices are typically:
    // 1. FFN weight matrices (w1, w2, w3) - These are often the largest
    // 2. Attention projection matrices (wq, wk, wv, wo)
    
    // Default SVD rank ratios (actual rank = ratio * min(rows, cols))
    public static class RankRatios
    {
        // Query, Key, Value projections (typically benefit from higher ranks)
        public const float WQ = 0.2f; // 20% of min dimension
        public const float WK = 0.2f;
        public const float WV = 0.2f;
        
        // Output projection (can often use lower rank)
        public const float WO = 0.25f;
        
        // Feed-forward networks (typically benefit most from SVD)
        public const float W1 = 0.15f; // Often has redundancy
        public const float W2 = 0.15f;
        public const float W3 = 0.15f;
    }
    
    // Determine if a matrix should use SVD based on its dimensions
    public static bool ShouldUseSvd(int rows, int cols, float rankRatio)
    {
        // Only apply SVD to large matrices
        if (rows < MinRowsForSvd || cols < MinColsForSvd)
            return false;
            
        if (rows * cols < MinElementsForSvd)
            return false;
            
        // Calculate effective rank
        int minDimension = Math.Min(rows, cols);
        int effectiveRank = (int)(minDimension * rankRatio);
        
        // Calculate theoretical speedup
        // Original: rows * cols multiplications
        // SVD: rank * (rows + cols) multiplications
        float originalOps = rows * cols;
        float svdOps = effectiveRank * (rows + cols);
        float speedupRatio = originalOps / svdOps;
        
        // Only use SVD if the theoretical speedup is significant
        return speedupRatio >= MinSpeedupRatio;
    }
}