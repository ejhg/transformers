using NUnit.Framework;

namespace llama;

[TestFixture]
public class MathOpsTests
{
    [Test]
    public void TestAdd () {
        double[] a = {
            1.0,
            2.0,
            3.0
        };
        double[] b = {
            4.0,
            5.0,
            6.0
        };
        double[] expected = {
            5.0,
            7.0,
            9.0
        };
        double[] result = MathOps.Add (a, b);
        Assert.AreEqual (expected, result);
    }

    [Test]
    public void TestDot () {
        double[] a = {
            1.0,
            2.0,
            3.0
        };
        double[] b = {
            4.0,
            5.0,
            6.0
        };
        double expected = 32.0; // 1*4 + 2*5 + 3*6
        double result = MathOps.Dot (a, b);
        Assert.AreEqual (expected, result);
    }

    [Test]
    public void TestMatrixVectorProduct () {
        double[,] matrix = new double[,] {
            {
                1,
                2,
                3
            }, {
                4,
                5,
                6
            }
        };
        double[] vector = {
            7,
            8,
            9
        };
        double[] expected = new double[] {
            50.0, // 1*7+2*8+3*9
            122.0 // 4*7+5*8+6*9
        };
        double[] result = MathOps.MatrixVectorProduct (matrix, vector);
        Assert.AreEqual (expected, result);
    }

    [Test]
    public void TestSoftmax () {
        double[] logits = {
            1.0,
            2.0,
            3.0
        };
        double[] result = MathOps.Softmax (logits);
        double maxLogit = logits.Max ();
        double[] expLogits = logits.Select (x => Math.Exp (x - maxLogit)).ToArray ();
        double sumExp = expLogits.Sum ();
        double[] expected = expLogits.Select (x => x / sumExp).ToArray ();
        Assert.AreEqual (expected, result);
    }

    [Test]
    public void TestGetRow () {
        double[,] matrix = new double[,] {
            {
                1,
                2,
                3
            }, {
                4,
                5,
                6
            }
        };
        double[] expected = {
            4,
            5,
            6
        };
        double[] result = MathOps.GetRow (matrix, 1);
        Assert.AreEqual (expected, result);
    }

    [Test]
    public void TestInitializeMatrix () {
        Random rand = new Random (0);
        int rows = 2;
        int cols = 3;
        double[,] matrix = MathOps.InitializeMatrix (rand, rows, cols);
        Assert.AreEqual (rows, matrix.GetLength (0));
        Assert.AreEqual (cols, matrix.GetLength (1));
        // Check that values are within the expected limit
        double limit = Math.Sqrt (6.0 / (rows + cols));
        for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            Assert.IsTrue (matrix[i, j] >= -limit && matrix[i, j] <= limit);
    }
}
