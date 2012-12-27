package soleun;

import static com.googlecode.javacv.cpp.opencv_core.*;
import static com.googlecode.javacv.cpp.opencv_imgproc.CV_BGR2GRAY;
import static com.googlecode.javacv.cpp.opencv_imgproc.cvCvtColor;
import static com.googlecode.javacv.cpp.opencv_objdetect.*;

import java.io.IOException;
import java.util.Iterator;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

import com.googlecode.javacv.cpp.opencv_core.CvMemStorage;
import com.googlecode.javacv.cpp.opencv_core.CvRect;
import com.googlecode.javacv.cpp.opencv_core.CvScalar;
import com.googlecode.javacv.cpp.opencv_core.CvSeq;
import com.googlecode.javacv.cpp.opencv_core.IplImage;
import com.googlecode.javacv.cpp.opencv_objdetect.CvHaarClassifierCascade;

import edu.vt.input.ImageInputFormat;
import edu.vt.output.ImageOutputFormat;
import edu.vt.io.Image;

public class FaceDetection extends Configured implements Tool {
	public static class Map extends Mapper<Text, Image, Text, Image> {

		// Create memory for calculations
		CvMemStorage storage = null;

		// Create a new Haar classifier
		CvHaarClassifierCascade classifier = null;

		@Override
		protected void setup(Context context) {
			// Allocate the memory storage
			storage = CvMemStorage.create();
			
			// Load the HaarClassifierCascade
			//classifier = new CvHaarClassifierCascade(cvLoad(classifierFile.getAbsolutePath()));
			classifier = new CvHaarClassifierCascade(cvLoad(context.getConfiguration().get("opencv.classifier")));
			
			// Make sure the cascade is loaded
			if (classifier.isNull()) {
				System.err.println("Error loading classifier file");
			}
		}

		@Override
		public void map(Text key, Image value, Context context)
				throws IOException, InterruptedException {
			
			// Clear the memory storage which was used before
            //cvClearMemStorage(storage);
			
			// We need a grayscale image in order to do the recognition, so we
	        // create a new image of the same size as the original one.
	        IplImage grayImage = IplImage.create(value.getImage().width(),
	                value.getImage().height(), IPL_DEPTH_8U, 1);
	 
	        // We convert the original image to grayscale.
	        cvCvtColor(value.getImage(), grayImage, CV_BGR2GRAY);
	 
	        // We detect the faces.
	        CvSeq faces = cvHaarDetectObjects(grayImage, classifier, storage, 1.5, 3, 0);
	 
	        // We iterate over the discovered faces and draw yellow rectangles
	        // around them.
	        for (int i = 0; i < faces.total(); i++) {
	            CvRect r = new CvRect(cvGetSeqElem(faces, i));
	            cvRectangle(value.getImage(), cvPoint(r.x(), r.y()),
	                    cvPoint(r.x() + r.width(), r.y() + r.height()),
	                    CvScalar.RED, 5, CV_AA, 0);
	        }
	        
	        context.write(key, value);

			/*if(!classifier.isNull()){
				// Detect the objects and store them in the sequence
				CvSeq faces = cvHaarDetectObjects(value.getImage(), classifier,
						storage, 1.1, 3, CV_HAAR_DO_CANNY_PRUNING);
				
				// Loop the number of faces found.  Draw red box around face.
				int total = faces.total();
				for (int i = 0; i < total; i++) {
					CvRect r = new CvRect(cvGetSeqElem(faces, i));
					int x = r.x(), y = r.y(), w = r.width(), h = r.height();
					String strRect = String.format("CvRect(%d,%d,%d,%d)", x, y, w, h);
					context.write(key, new Text(strRect));
				}
			}*/
		}
	}

	public static class Reduce extends Reducer<Text, Image, Text, Image> {

		@Override
		public void reduce(Text key, Iterable<Image> values, Context context)
				throws IOException, InterruptedException {

			Iterator<Image> it = values.iterator();
			while (it.hasNext()) {
				context.write(key, it.next());
			}
		}
	}

	public int run(String[] args) throws Exception {
		// Set various configuration settings
		Configuration conf = getConf();
		conf.setInt("mapreduce.imagerecordreader.windowsizepercent", 100);
		conf.setInt("mapreduce.imagerecordreader.borderPixel", 0);
		conf.setInt("mapreduce.imagerecordreader.iscolor", 1);
		conf.setStrings("opencv.classifier", new String(args[0]));

		// Create job
		Job job = new Job(conf);

		// Specify various job-specific parameters
		job.setJarByClass(FaceDetection.class);
		job.setJobName("FaceDetection");

		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(Image.class);

		job.setMapperClass(Map.class);
		job.setReducerClass(Reduce.class);

		job.setInputFormatClass(ImageInputFormat.class);
		job.setOutputFormatClass(ImageOutputFormat.class);

		FileInputFormat.addInputPath(job, new Path(args[1]));
		FileOutputFormat.setOutputPath(job, new Path(args[2]));

		return job.waitForCompletion(true) ? 0 : 1;
	}

	public static void main(String[] args) throws Exception {
		long t = cvGetTickCount();
		int res = ToolRunner.run(new Configuration(), new FaceDetection(), args);
		t = cvGetTickCount() - t;
		System.out.println("Time: " + (t/cvGetTickFrequency()/1000000) + "sec");
		System.exit(res);
	}
}