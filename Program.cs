
namespace teste;

public class Program
{
    static void Main(string[] args)
    {
        //CoffeeBoxDetector.DetectTextWithOCR(Models._predictSingleImage, Path.Combine(Models._assetsPath, "output"));

        CoffeeBoxDetector.DetectObjectsMatchingTemplates(Models._predictSingleImage);
    }
}
