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

    public static bool IsRegionFree(Mat mask, Rectangle region, int overlapThreshold = 30)
    {
        if (region.X < -overlapThreshold || region.Y < -overlapThreshold ||
            region.X + region.Width > mask.Cols + overlapThreshold ||
            region.Y + region.Height > mask.Rows + overlapThreshold ||
            region.Width <= 0 || region.Height <= 0)
        {
            return false; // Region invalid
        }

        foreach (var existingRegion in Utilities.DetectedRegions)
        {
            if (IsOverlapping(existingRegion, region, overlapThreshold))
                return false; // Dacă există suprapunere peste limită
        }

        return true;
    }

    private static bool IsOverlapping(Rectangle existing, Rectangle newRegion, int threshold)
    {
        return !(newRegion.Right < existing.Left - threshold ||
                 newRegion.Left > existing.Right + threshold ||
                 newRegion.Bottom < existing.Top - threshold ||
                 newRegion.Top > existing.Bottom + threshold);
    }


    public static void ExcludeRegionFromMask(Mat mask, Rectangle region)
    {
        CvInvoke.Rectangle(mask, region, new MCvScalar(255), -1); // Mark the region with white (255) to indicate exclusion
    }

    public static Mat[] RotateImageMultipleTimes(Mat subImage, int rotations = 7, double angleStep = 45)
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
