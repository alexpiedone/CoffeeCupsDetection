using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.OCR;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using System.Drawing;
using teste;

public static class CoffeeBoxDetector
{
    private const double AreaThreshold = 1000; // Minimum area to consider as a box
    private const double AspectRatioThreshold = 0.7; // Aspect ratio tolerance for boxes
    static readonly string[] templatePaths = Directory.GetFiles(Models._samplesFolder, "*.jpg");
    public static readonly string[] imagesToPredict = Directory.GetFiles(Models._imagesFolder, "*.jpg");



    public static void DetectObjectsMatchingTemplates(string path)
    {
        Mat image = CvInvoke.Imread(path, ImreadModes.Unchanged);

        var validScales = new List<double> { 1.75, 2.0 };
        Mat exclusionMask = Utilities.CreateExclusionMask(image);

        foreach (var templatePath in templatePaths)
        {
            Mat template = CvInvoke.Imread(templatePath, ImreadModes.Color);

            foreach (var scale in validScales)
            {
                Mat resizedTemplate = Utilities.ResizeTemplate(template, scale);

                Mat result = new Mat();
                CvInvoke.MatchTemplate(image, resizedTemplate, result, Emgu.CV.CvEnum.TemplateMatchingType.CcoeffNormed);
                float[,] resultData = result.GetData() as float[,];

                for (int y = 0; y < result.Rows; y++)
                {
                    for (int x = 0; x < result.Cols; x++)
                    {
                        if (resultData[y, x] > 0.55)
                        {
                            Point matchPoint = new Point(x - 5, y - 5);
                            Rectangle matchRect = new Rectangle(matchPoint, new Size(resizedTemplate.Width +10, resizedTemplate.Height +10));

                            if (Utilities.IsRegionFree(exclusionMask, matchRect))
                            {
                                Mat subImage = new Mat(image, matchRect);

                                string templateName = Path.GetFileNameWithoutExtension(templatePath);

                                Utilities.ExcludeRegionFromMask(exclusionMask, matchRect);
                               
                                CvInvoke.Rectangle(image, matchRect, new MCvScalar(0, 255, 0), 2);

                                var detectedText = DetectTextWithOCR(subImage);
                                if (!string.IsNullOrEmpty(detectedText))
                                {
                                    Point textPosition = new Point(matchRect.X, matchRect.Y + 15);
                                    CvInvoke.PutText(image, detectedText, textPosition, FontFace.HersheySimplex, 0.8, new MCvScalar(50, 120, 255), 2);
                                }

                            }
                        }
                    }
                }
            }
        }
        image.Save(Path.Combine(Models._outputFolder, string.Concat(Path.GetFileNameWithoutExtension(path), "_output.jpg")));
    }


    public static string DetectTextWithOCR(Mat image, string? imagePath = null)
    {
        List<string> expectedOutcomes = new List<string> { "VOLTESSO", "ORAFIO", "BIANCO", "PICOLLO", "DOLCE" };
        if (image == null)
            image = CvInvoke.Imread(imagePath, ImreadModes.Unchanged);

        string PerformOCR(Mat img)
        {
            using (Tesseract tesseract = new Tesseract("D:\\Practice\\teste\\tessdata", "eng", OcrEngineMode.TesseractLstmCombined))
            {
                //piedone-dupa teste am observat ca cele mai bune rezultate le a obtinut PageSegMode.Auto(le am testat pe toate aproape)
                tesseract.PageSegMode = PageSegMode.Auto;
                tesseract.SetVariable("tessedit_char_whitelist", "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz  ");
                tesseract.SetVariable("classify_enable_learning", "0");
                tesseract.SetImage(img);
                tesseract.Recognize();
                return tesseract.GetUTF8Text().Trim();
            }
        }

        Mat grayImage = new Mat();
        CvInvoke.CvtColor(image, grayImage, ColorConversion.Bgr2Gray);
        //piedone-filtrul gausian ajuta semnificativ, testat cu valori de la Size(1, 1) la Size(3, 3)
        CvInvoke.GaussianBlur(grayImage, grayImage, new Size(3, 3), 0);
        CvInvoke.Threshold(grayImage, grayImage, 0, 255, ThresholdType.Otsu | ThresholdType.Binary);
        Mat kernel = CvInvoke.GetStructuringElement(ElementShape.Rectangle, new Size(1, 1), new Point(-1, -1));
          //piedone-dupa teste am observat ca cele mai bune rezultate le a obtinut MorphOp.Dilate
        CvInvoke.MorphologyEx(grayImage, grayImage, MorphOp.Dilate, kernel, new Point(-1, -1), 1, BorderType.Default, new MCvScalar());

        string label = PerformOCR(grayImage);
        if (!string.IsNullOrEmpty(label) && expectedOutcomes.Any(expected => label.Contains(expected, StringComparison.OrdinalIgnoreCase)))
        {
            return label;
        }

        Mat[] rotatedImages = Utilities.RotateImageMultipleTimes(grayImage);
        foreach (Mat rotatedImage in rotatedImages)
        {
            label = PerformOCR(rotatedImage);
            if (!string.IsNullOrEmpty(label) && expectedOutcomes.Any(expected => label.Contains(expected, StringComparison.OrdinalIgnoreCase)))
            {
                return label;
            }
        }

        return string.Empty;
    }

    //Environment.SetEnvironmentVariable("TESSDATA_PREFIX", "./tessdata");
    //   var psmModes = new[] {
    //       PageSegMode.Auto,           // --psm 3
    //       PageSegMode.SingleColumn,   // --psm 4
    //       PageSegMode.CircleWord,   // --psm 4
    //       PageSegMode.SingleBlockVertText,   // --psm 4
    //       PageSegMode.SparseText,   // --psm 4
    //       PageSegMode.SingleBlock,    // --psm 6
    //       PageSegMode.SingleLine      // --psm 7
    //   };

    //   var results = new Dictionary<PageSegMode, string>();
    //   foreach (var mode in psmModes)
    //   {
    //       using (Tesseract tesseract = new Tesseract("D:\\Practice\\teste\\tessdata", "eng", OcrEngineMode.TesseractLstmCombined))
    //       {
    //           tesseract.PageSegMode = mode;
    //           tesseract.SetVariable("tessedit_char_whitelist", "ABCDEFGHIJKLMNOPQRSTUVWXYZ ");
    //           tesseract.SetImage(grayImage);
    //           tesseract.Recognize();

    //           string label = tesseract.GetUTF8Text().Trim();
    //           results[mode] = label;
    //       }
    //   }


    public static Mat DeskewImage(Mat grayImage)
    {
        Mat edges = new Mat();
        CvInvoke.Canny(grayImage, edges, 50, 150, 3);

        LineSegment2D[] lines = CvInvoke.HoughLinesP(edges, 1, Math.PI / 180, 100, 50, 10);
        double angle = 0;
        int count = 0;

        foreach (var line in lines)
        {
            double currentAngle = Math.Atan2(line.P2.Y - line.P1.Y, line.P2.X - line.P1.X) * (180 / Math.PI);
            if (Math.Abs(currentAngle) > 5 && Math.Abs(currentAngle) < 85) // Ignore near-horizontal lines
            {
                angle += currentAngle;
                count++;
            }
        }

        if (count > 0) angle /= count;

        Mat rotated = new Mat();
        Mat rotationMatrix = new Mat();
        CvInvoke.GetRotationMatrix2D(new PointF(grayImage.Width / 2, grayImage.Height / 2), angle, 1, rotationMatrix);
        CvInvoke.WarpAffine(grayImage, rotated, rotationMatrix, grayImage.Size);
        return rotated;
    }

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


}