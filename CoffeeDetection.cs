using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.OCR;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using System.Drawing;
using System.IO;
using System.Security.Cryptography.X509Certificates;
using static System.Net.Mime.MediaTypeNames;

public static class CoffeeBoxDetector
{
    private const double AreaThreshold = 1000; // Minimum area to consider as a box
    private const double AspectRatioThreshold = 0.7; // Aspect ratio tolerance for boxes
    static readonly string[] templatePaths = Directory.GetFiles(Models._samplesFolder, "*.jpg");

    public static void DetectCoffeeBoxesByContour(string imagePath, string outputPath)
    {
        // Load the image
        Mat image = CvInvoke.Imread(imagePath, ImreadModes.Color);

        // Convert the image to grayscale for contour detection
        Mat grayImage = new Mat();
        CvInvoke.CvtColor(image, grayImage, ColorConversion.Bgr2Gray);

        // Apply Gaussian blur to reduce noise
        CvInvoke.GaussianBlur(grayImage, grayImage, new Size(5, 5), 0);

        // Perform edge detection
        Mat edges = new Mat();
        CvInvoke.Canny(grayImage, edges, 50, 150);

        // Find contours in the edges image
        VectorOfVectorOfPoint contours = new VectorOfVectorOfPoint();
        Mat hierarchy = new Mat();
        CvInvoke.FindContours(edges, contours, hierarchy, RetrType.List, ChainApproxMethod.ChainApproxSimple);

        // Set the TESSDATA_PREFIX environment variable
        Environment.SetEnvironmentVariable("TESSDATA_PREFIX", "./tessdata");

        // Initialize Tesseract OCR
        using (Tesseract tesseract = new Tesseract("D:\\Practice\\teste\\tessdata", "eng", OcrEngineMode.TesseractLstmCombined))
        {
            tesseract.PageSegMode = PageSegMode.SingleLine;

            // Process each contour
            for (int i = 0; i < contours.Size; i++)
            {
                VectorOfPoint contour = contours[i];
                double area = CvInvoke.ContourArea(contour);

                // Filter out small contours
                if (area < AreaThreshold)
                    continue;

                // Approximate the contour to a polygon
                VectorOfPoint approxContour = new VectorOfPoint();
                CvInvoke.ApproxPolyDP(contour, approxContour, CvInvoke.ArcLength(contour, true) * 0.02, true);

                // Check if the contour has 4 vertices (likely a rectangle)
                if (approxContour.Size == 4)
                {
                    // Calculate aspect ratio to ensure it's a rectangle
                    Rectangle boundingRect = CvInvoke.BoundingRectangle(approxContour);
                    double aspectRatio = (double)boundingRect.Width / boundingRect.Height;

                    if (aspectRatio >= AspectRatioThreshold && aspectRatio <= 1.0 / AspectRatioThreshold)
                    {
                        // Draw the bounding box
                        CvInvoke.Rectangle(image, boundingRect, new MCvScalar(0, 255, 0), 2);

                        // Extract the region of interest (ROI) for OCR
                        Mat roi = new Mat(image, boundingRect);

                        // Perform OCR on the ROI
                        tesseract.SetImage(roi);
                        tesseract.Recognize();
                        string label = tesseract.GetUTF8Text().Trim();
                        if (!string.IsNullOrEmpty(label))
                        {
                            Console.WriteLine("Textul extras: " + label);
                        }
                        // Annotate the box with the extracted label
                        if (!string.IsNullOrEmpty(label))
                        {
                            CvInvoke.PutText(image, label, new Point(boundingRect.X, boundingRect.Y - 10),
                                FontFace.HersheySimplex, 0.7, new MCvScalar(0, 0, 255), 2);
                        }
                    }
                }
            }
        }

        // Save the output image
        string outputFilePath = Path.Combine(outputPath, Path.GetFileName(imagePath));
        image.Save(outputFilePath);
    }


    public static void DetectObjectsMatchingTemplates(string path)
    {
        Mat image = CvInvoke.Imread(path, ImreadModes.Color);

        var validScales = new List<double> { 1.75, 2.0 };
        Mat exclusionMask = CreateExclusionMask(image);

        foreach (var templatePath in templatePaths)
        {
            Mat template = CvInvoke.Imread(templatePath, ImreadModes.Color);

            foreach (var scale in validScales)
            {
                Mat resizedTemplate = ResizeTemplate(template, scale);

                Mat result = new Mat();
                CvInvoke.MatchTemplate(image, resizedTemplate, result, Emgu.CV.CvEnum.TemplateMatchingType.CcoeffNormed);
                float[,] resultData = result.GetData() as float[,];

                for (int y = 0; y < result.Rows; y++)
                {
                    for (int x = 0; x < result.Cols; x++)
                    {
                        if (resultData[y, x] > 0.55)
                        {
                            //Console.WriteLine($"Match found at  scale {scale}");
                            Point matchPoint = new Point(x, y - 30);
                            Rectangle matchRect = new Rectangle(matchPoint, new Size(resizedTemplate.Width, resizedTemplate.Height + 70));

                            if (IsRegionFree(exclusionMask, matchRect))
                            {
                                Mat subImage = new Mat(image, matchRect);

                                string templateName = Path.GetFileNameWithoutExtension(templatePath);

                                ExcludeRegionFromMask(exclusionMask, matchRect);
                                DrawMatch(image, matchRect, templatePath);
                                //foreach (var rotatedImage in RotateImageMultipleTimes(subImage))
                                //{
                                //    DetectTextWithOCR(rotatedImage);
                                //}
                                DetectTextWithOCR(subImage);
                            }
                        }
                    }
                }
            }
        }

        image.Save(string.Concat(Path.GetFileNameWithoutExtension(path), "_output.jpg"));
    }

    public static Mat[] RotateImageMultipleTimes(Mat subImage, int rotations = 5, double angleStep = 45)
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

    public static void DetectTextWithOCR(Mat image, string? imagePath = null)
    {
        List<string> expectedOutcomes = new List<string> { "VOLTESSO", "ORAFIO", "BIANCO", "PICOLLO", "DOLCE" };
        // Load the image
        if (image == null)
            image = CvInvoke.Imread(imagePath, ImreadModes.Color);

        // Convert the image to grayscale for better OCR results
        Mat grayImage = new Mat();
        CvInvoke.CvtColor(image, grayImage, ColorConversion.Bgr2Gray);

        // Optionally, apply some preprocessing (Gaussian blur) to reduce noise
        CvInvoke.GaussianBlur(grayImage, grayImage, new Size(5, 5), 0);
        CvInvoke.Threshold(grayImage, grayImage, 0, 255, ThresholdType.Otsu | ThresholdType.Binary);

        // Set the TESSDATA_PREFIX environment variable (if necessary)
        Environment.SetEnvironmentVariable("TESSDATA_PREFIX", "./tessdata");

        // Initialize Tesseract OCR
        using (Tesseract tesseract = new Tesseract("D:\\Practice\\teste\\tessdata", "eng", OcrEngineMode.Default))
        {
            tesseract.PageSegMode = PageSegMode.Auto;
            tesseract.SetVariable("tessedit_char_whitelist", "ABCDEFGHIJKLMNOPQRSTUVWXYZ");
            //abcdefghijklmnopqrstuvwxyz
            // Set the image for OCR
            tesseract.SetImage(grayImage);

            // Perform OCR on the entire image
            tesseract.Recognize();

            // Get the recognized text
            string label = tesseract.GetUTF8Text().Trim();

            if (!string.IsNullOrEmpty(label))
            {
                expectedOutcomes.ForEach(expectedOutcomes =>
                {
                    if (label.Contains(expectedOutcomes))
                    {
                        Console.WriteLine("Textul extras: " + label);
                    }
                });
            }
        }

        // Optionally, save the output image (if you wish to save it with annotations)
        //string outputFilePath = Path.Combine(outputPath, Path.GetFileName(imagePath));
        //image.Save(outputFilePath);
    }
    //public static Mat DeskewImage(Mat grayImage)
    //{
    //    Mat edges = new Mat();
    //    CvInvoke.Canny(grayImage, edges, 50, 150, 3);

    //    LineSegment2D[] lines = CvInvoke.HoughLinesP(edges, 1, Math.PI / 180, 100, 50, 10);
    //    double angle = 0;
    //    int count = 0;

    //    foreach (var line in lines)
    //    {
    //        double currentAngle = Math.Atan2(line.P2.Y - line.P1.Y, line.P2.X - line.P1.X) * (180 / Math.PI);
    //        if (Math.Abs(currentAngle) > 5 && Math.Abs(currentAngle) < 85) // Ignore near-horizontal lines
    //        {
    //            angle += currentAngle;
    //            count++;
    //        }
    //    }

    //    if (count > 0) angle /= count;

    //    Mat rotated = new Mat();
    //    CvInvoke.WarpAffine(grayImage, rotated, CvInvoke.GetRotationMatrix2D(new PointF(grayImage.Width / 2, grayImage.Height / 2), angle, 1), grayImage.Size);
    //    return rotated;
    //}







    private static Mat CreateExclusionMask(Mat image)
    {
        Mat exclusionMask = new Mat(image.Size, DepthType.Cv8U, 1);
        exclusionMask.SetTo(new MCvScalar(0));
        return exclusionMask;
    }
    private static Mat ResizeTemplate(Mat template, double scale)
    {
        Mat resizedTemplate = new Mat();
        CvInvoke.Resize(template, resizedTemplate, new Size(0, 0), scale, scale, Emgu.CV.CvEnum.Inter.Linear);
        return resizedTemplate;
    }

    private static void DrawMatch(Mat image, Rectangle matchRect, string templatePath)
    {
        CvInvoke.Rectangle(image, matchRect, new MCvScalar(0, 255, 0), 2);
        Point textPosition = new Point(matchRect.X, matchRect.Y + 10);
        //CvInvoke.PutText(image, Path.GetFileName(templatePath), textPosition, FontFace.HersheySimplex, 0.7, new MCvScalar(50, 120, 255), 2);
    }

    private static bool IsRegionFree(Mat mask, Rectangle region)
    {
        Mat roi = new Mat(mask, region);
        MCvScalar mean = CvInvoke.Mean(roi); // Check if the region is empty (i.e., value is 0)
        return mean.V0 == 0; // If the mean value is 0, the region is free
    }

    private static void ExcludeRegionFromMask(Mat mask, Rectangle region)
    {
        CvInvoke.Rectangle(mask, region, new MCvScalar(255), -1); // Mark the region with white (255) to indicate exclusion
    }
}