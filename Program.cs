
namespace teste;

public class Program
{
    static void Main(string[] args)
    {
        //CoffeeBoxDetector.DetectTextWithOCR(Models._predictSingleImage, Path.Combine(Models._assetsPath, "output"));

        foreach (var file in CoffeeBoxDetector.imagesToPredict)
        {
            CoffeeBoxDetector.DetectObjectsMatchingTemplates(file);
        }
    }
}
