namespace mingpt3;

public static class Dropout
{
    static Random rand = new Random ();

    public static Matrix Apply (Matrix input, double dropoutRate, bool training = true) {
        if (!training || dropoutRate <= 0.0) {
            return input;
        }

        var result = new Matrix (input.Rows, input.Cols);
        double scale = 1.0 / (1.0 - dropoutRate);

        for (int i = 0; i < input.Rows; i++) {
            for (int j = 0; j < input.Cols; j++) {
                if (rand.NextDouble () < dropoutRate) {
                    result.Data[i, j] = 0.0;
                } else {
                    result.Data[i, j] = input.Data[i, j] * scale;
                }
            }
        }

        return result;
    }

    public static Matrix ApplyInPlace (Matrix input, double dropoutRate, bool training = true) {
        if (!training || dropoutRate <= 0.0) {
            return input;
        }

        double scale = 1.0 / (1.0 - dropoutRate);

        for (int i = 0; i < input.Rows; i++) {
            for (int j = 0; j < input.Cols; j++) {
                if (rand.NextDouble () < dropoutRate) {
                    input.Data[i, j] = 0.0;
                } else {
                    input.Data[i, j] *= scale;
                }
            }
        }

        return input;
    }
}
