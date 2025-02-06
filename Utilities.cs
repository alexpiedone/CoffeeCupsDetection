using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV;
using System.Drawing;

namespace teste;

public class Utilities
{
    public static Mat CreateExclusionMask(Mat image)
    {
        Mat exclusionMask = new Mat(image.Size, DepthType.Cv8U, 1);
        exclusionMask.SetTo(new MCvScalar(0));
        return exclusionMask;
    }

    public static Mat ResizeTemplate(Mat template, double scale)
    {
        Mat resizedTemplate = new Mat();
        CvInvoke.Resize(template, resizedTemplate, new Size(0, 0), scale, scale, Emgu.CV.CvEnum.Inter.Linear);
        return resizedTemplate;
    }

    public static bool IsRegionFree(Mat mask, Rectangle region, int overlapTolerance = 30)
    {
        if (region.X < 0 || region.Y < 0 ||
            region.X + region.Width > mask.Cols ||
            region.Y + region.Height > mask.Rows ||
            region.Width <= 0 || region.Height <= 0)
        {
            return false; // Region invalid
        }

        Mat roi = new Mat(mask, region);
        MCvScalar mean = CvInvoke.Mean(roi);

        // Calculăm procentajul pixelilor liberi (valoare 0)
        double freeRatio = mean.V0 / 255.0;

        // Permitem overlap dacă cel puțin 30% din regiune este liberă
        return freeRatio < (overlapTolerance / 100.0);
    }

    public static void ExcludeRegionFromMask(Mat mask, Rectangle region)
    {
        CvInvoke.Rectangle(mask, region, new MCvScalar(255), -1); // Mark the region with white (255) to indicate exclusion
    }

    public static Mat[] RotateImageMultipleTimes(Mat subImage, int rotations = 11, double angleStep = 30)
    {
        Mat[] rotatedImages = new Mat[rotations];
        Size size = subImage.Size;
        PointF center = new PointF(size.Width / 2f, size.Height / 2f);

        for (int i = 0; i < rotations; i++)
        {
            double angle = angleStep * (i + 1);
            Mat rotationMatrix = new Mat();
            CvInvoke.GetRotationMatrix2D(center, angle, 1.0, rotationMatrix);
            Mat rotated = new Mat();
            CvInvoke.WarpAffine(subImage, rotated, rotationMatrix, size, Inter.Linear, Warp.Default, BorderType.Constant, new MCvScalar(0, 0, 0));
            rotatedImages[i] = rotated;
        }

        return rotatedImages;
    }

}
