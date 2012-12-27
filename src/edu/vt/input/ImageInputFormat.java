package edu.vt.input;

import java.io.IOException;

import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.InputSplit;
import org.apache.hadoop.mapreduce.JobContext;
import org.apache.hadoop.mapreduce.RecordReader;
import org.apache.hadoop.mapreduce.TaskAttemptContext;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;

import edu.vt.io.Image;

public class ImageInputFormat extends FileInputFormat<Text, Image> {

	@Override
	public RecordReader<Text, Image> createRecordReader(InputSplit split,
			TaskAttemptContext context) throws IOException,
			InterruptedException {
		return new ImageRecordReader();
	}

	@Override
	protected boolean isSplitable(JobContext context, Path file) {
		return false;
	}
}