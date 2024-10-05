using System;

class ContinuousHopfieldNetwork
{
    static void Main()
    {
        int N = 5;
        double[,] W = new double[N, N];
        double[][] patterns = {
            new double[] { 1, -1, 1, -1, 1 },
            new double[] { -1, 1, -1, 1, -1 }
        };

        // Initialize weights
        for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            if (i != j)
                foreach (var p in patterns)
                    W[i, j] += p[i] * p[j] / N;

        double[] x = { 0.5, -0.3, 0.1, -0.7, 0.2 };

        // Update network
        for (int t = 0; t < 10; t++)
        {
            double[] x_new = new double[N];
            for (int i = 0; i < N; i++)
            {
                double u = 0;
                for (int j = 0; j < N; j++)
                    u += W[i, j] * x[j];
                x_new[i] = Math.Tanh(u);
            }
            x = x_new;
            Console.WriteLine($"t={t}: [{string.Join(", ", x)}]");
        }
    }
}
