class HopfieldNetwork
{
    int size;
    double[,] weights;

    public HopfieldNetwork (int size) {
        this.size = size;
        weights = new double[size, size];
    }

    public void Train (int[][] patterns) {
        foreach (var pattern in patterns)
            for (int i = 0; i < size; i++)
            for (int j = 0; j < size; j++)
                if (i != j)
                    weights[i, j] += pattern[i] * pattern[j];
    }

    public int[] Recall (int[] input, int steps = 10) {
        int[] output = (int[])input.Clone ();
        for (int s = 0; s < steps; s++)
        for (int i = 0; i < size; i++) {
            double sum = 0;
            for (int j = 0; j < size; j++)
                sum += weights[i, j] * output[j];
            output[i] = sum >= 0 ? 1 : -1;
        }

        return output;
    }

    static void Main () {
        int size = 4;
        var net = new HopfieldNetwork (size);

        int[] pattern1 = {
            1,
            -1,
            1,
            -1
        };
        int[] pattern2 = {
            -1,
            1,
            -1,
            1
        };
        net.Train (new int[][] {
            pattern1,
            pattern2
        });

        int[] noisyPattern = {
            1,
            1,
            1,
            -1
        };
        int[] recalledPattern = net.Recall (noisyPattern);

        Console.WriteLine ("Recalled Pattern: " + string.Join (", ", recalledPattern));
    }
}
