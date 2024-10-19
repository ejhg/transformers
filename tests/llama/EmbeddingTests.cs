using NUnit.Framework;

namespace llama;

[TestFixture]
public class EmbeddingTests
{
    [Test]
    public void TestForward()
    {
        int vocabSize = 10;
        int embedSize = 5;
        Random rand = new Random(0);
        Embedding embedding = new Embedding(vocabSize, embedSize, rand);
        int token = 3;
        double[] expected = new double[embedSize];
        for (int i = 0; i < embedSize; i++)
        {
            expected[i] = embedding.Weights[token, i];
        }
        double[] result = embedding.Forward(token);
        Assert.AreEqual(expected, result);
    }

    [Test]
    public void TestBackward()
    {
        int vocabSize = 10;
        int embedSize = 5;
        Random rand = new Random(0);
        Embedding embedding = new Embedding(vocabSize, embedSize, rand);
        int token = 3;
        double[] grad = { 1.0, 1.0, 1.0, 1.0, 1.0 };
        embedding.Backward(token, grad);
        for (int i = 0; i < embedSize; i++)
        {
            Assert.AreEqual(grad[i], embedding.Gradients[token, i]);
        }
    }
}
